import os
import json
import shutil
import base64
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Google API Key is not set in environment variables.")

app = FastAPI(title="Hayy Ibtikar Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعداد النماذج
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview", google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=API_KEY)

# ==========================================
# 1. إعداد الذاكرة المتعددة لأقسام حي ابتكار
# ==========================================
CATEGORIES = {
    "patents": {"file": "data/patents.json", "index": "faiss_patents"},
    "research": {"file": "data/research.json", "index": "faiss_research"},
    "startups": {"file": "data/startups.json", "index": "faiss_startups"},
    "investments": {"file": "data/investments.json", "index": "faiss_investments"}
}

vectorstores = {}
retrievers = {}

def initialize_category(category_name):
    """دالة لإنشاء أو تحميل قاعدة البيانات لكل قسم"""
    file_path = CATEGORIES[category_name]["file"]
    index_path = CATEGORIES[category_name]["index"]
    
    os.makedirs("data", exist_ok=True)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            
    if data:
        documents = [Document(page_content=json.dumps(item, ensure_ascii=False)) for item in data]
        vs = FAISS.from_documents(documents, embeddings)
    else:
        vs = FAISS.from_texts(["لا توجد بيانات حاليا في هذا القسم لحي ابتكار."], embeddings)
        
    vs.save_local(index_path)
    vectorstores[category_name] = vs
    retrievers[category_name] = vs.as_retriever(search_kwargs={"k": 15})

@app.on_event("startup")
async def startup_event():
    print("جاري تحميل قواعد بيانات حي ابتكار المستقلة...")
    for cat in CATEGORIES:
        initialize_category(cat)
    print("تم تحميل جميع الفهارس بنجاح!")


# ==========================================
# 2. إعداد سلسلة المحادثة (Prompt & Chain)
# ==========================================
template = """أنت مساعد ذكي لمنصة حي ابتكار. اتبع هذه القواعد بصرامة:
1. أجب بنفس لغة سؤال المستخدم بالضبط (إذا سأل بالعربية أجب بالعربية فقط، وإذا سأل بالإنجليزية أجب بالإنجليزية فقط).
2. يُمنع منعاً باتاً خلط اللغتين في نفس الإجابة، يُستثنى من ذلك فقط المصطلحات التقنية المعقدة أو أسماء الشركات والتقنيات التي ليس لها ترجمة.
3. إذا كان السؤال شخصياً أو مبنياً على حوار سابق، استخرج الإجابة من قسم <تاريخ_المحادثة>.
4. إذا كان السؤال عن الحي (براءات، أبحاث، شركات، استثمارات)، استخرج الإجابة من قسم <سياق_المعلومات>.
5. لا تقل "السياق لا يحتوي على معلومات" إذا كانت الإجابة موجودة في تاريخ المحادثة.

<تاريخ_المحادثة>
{history}
</تاريخ_المحادثة>

<سياق_المعلومات>
{context}
</سياق_المعلومات>

السؤال الحالي: {question}
الإجابة:"""

prompt = PromptTemplate.from_template(template)
qa_chain = prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class ChatRequest(BaseModel):
    message: str
    history: str = "" 

# ==========================================
# 2.5. إعداد الوكيل الموجه (Router Agent)
# ==========================================
router_template = """أنت وكيل توجيه (Router Agent) في منصة حي ابتكار.
مهمتك هي تحليل سؤال المستخدم وتحديد القسم الأنسب للإجابة عليه بدقة، اختر قسماً واحداً فقط من القائمة التالية:
- patents (إذا كان السؤال عن براءات الاختراع، التقنيات، والملكيات الفكرية)
- research (إذا كان السؤال عن الأبحاث، الأوراق العلمية، والمراكز)
- startups (إذا كان السؤال عن الشركات الناشئة، رواد الأعمال، والمشاريع)
- investments (إذا كان السؤال عن الفرص الاستثمارية، التمويل، والعوائد المالية)
- all (إذا كان السؤال عاماً عن الحي ككل أو يشمل عدة أقسام، أو مجرد ترحيب)

سؤال المستخدم: {question}

أجب بكلمة واحدة فقط باللغة الإنجليزية من الكلمات المذكورة أعلاه (patents, research, startups, investments, all).
القسم المختار:"""

router_prompt = PromptTemplate.from_template(router_template)
# نستخدم نفس النموذج اللغوي llm ليكون هو العقل الموجه
router_chain = router_prompt | llm | StrOutputParser()


# ==========================================
# 3. مسار الدردشة (مزود بالوكيل الموجه)
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. مرحلة التوجيه: سؤال الوكيل الموجه عن القسم المناسب
        category = router_chain.invoke({"question": request.message}).strip().lower()
        print(f"الوكيل الموجه قرر تحويل السؤال إلى قسم: {category}") # سيظهر هذا في سجلات Render
        
        # 2. مرحلة الاسترجاع: البحث في القسم المحدد فقط
        all_docs = []
        if category in retrievers:
            # إذا اختار قسماً محدداً، نبحث فيه فقط
            docs = retrievers[category].invoke(request.message)
            all_docs.extend(docs)
        else:
            # إذا اختار 'all' أو لم يفهم السؤال، نبحث في كل الأقسام احتياطياً
            for cat in retrievers:
                docs = retrievers[cat].invoke(request.message)
                all_docs.extend(docs)
                
        # نأخذ أفضل 15 نتيجة لضمان السياق العميق
        context_text = format_docs(all_docs[:15])
        
        # 3. مرحلة التوليد: صياغة الإجابة النهائية
        response = qa_chain.invoke({
            "context": context_text,
            "history": request.history,
            "question": request.message
        })
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 4. مسار رفع البيانات للأقسام
# ==========================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Haider414/patents-chatbot")

@app.post("/upload-data")
async def upload_data(category: str = Form(...), file: UploadFile = File(...)):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="قسم غير صالح. الرجاء اختيار قسم صحيح.")
        
    try:
        content_bytes = await file.read()
        file_content_str = content_bytes.decode("utf-8")
        new_data = json.loads(file_content_str)
        
        file_path = CATEGORIES[category]["file"]
        
        # قراءة البيانات الحالية للقسم
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            
        combined_data = existing_data + (new_data if isinstance(new_data, list) else [new_data])
        
        # 1. حفظ الملف في الذاكرة محلياً
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
            
        # 2. تحديث فهرس القسم المحدد فوراً
        initialize_category(category)
        
        # 3. الرفع التلقائي إلى GitHub لضمان حفظ البيانات دائماً
        if GITHUB_TOKEN and GITHUB_REPO:
            try:
                url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
                headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                
                # جلب رقم SHA للملف الحالي إذا كان موجوداً
                get_response = requests.get(url, headers=headers)
                sha = ""
                if get_response.status_code == 200:
                    sha = get_response.json().get("sha", "")
                
                # تجهيز البيانات للرفع
                updated_content = json.dumps(combined_data, ensure_ascii=False, indent=4)
                encoded_content = base64.b64encode(updated_content.encode("utf-8")).decode("utf-8")
                
                put_data = {
                    "message": f"Auto-update {category} data via API",
                    "content": encoded_content,
                    "branch": "main"
                }
                if sha:
                    put_data["sha"] = sha
                    
                requests.put(url, headers=headers, json=put_data)
            except Exception as gh_error:
                print(f"خطأ في المزامنة مع جيت هب: {gh_error}")

        return {"status": "success", "message": f"تم تحديث بيانات قسم {category} وحفظها بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))