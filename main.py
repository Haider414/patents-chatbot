import os
import json
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

# ==========================================
# 0. إعداد النماذج
# ==========================================
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview", google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=API_KEY)

# ==========================================
# 1. إعداد الذاكرة المتعددة والهيكل المتقدم
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
        documents = []
        for item in data:
            # دمج النصوص العربية والإنجليزية لتسهيل الفهم على النموذج
            combined_text = (
                f"العنوان (Title): {item.get('title_ar', item.get('title', ''))} | {item.get('title_en', '')}\n"
                f"الحالة (Status): {item.get('status', 'غير محدد')}\n"
                f"التفاصيل (Details AR): {item.get('text_ar', item.get('description', item.get('abstract', '')))}\n"
                f"التفاصيل (Details EN): {item.get('text_en', '')}"
            )
            
            # حفظ البيانات الوصفية بشكل منفصل
            meta = {
                "id": item.get("doc_id", item.get("id", "N/A")),
                "url": item.get("official_url", "N/A")
            }
            documents.append(Document(page_content=combined_text, metadata=meta))
            
        vs = FAISS.from_documents(documents, embeddings)
    else:
        vs = FAISS.from_texts(["لا توجد بيانات حاليا في هذا القسم لحي ابتكار."], embeddings)
        
    vs.save_local(index_path)
    vectorstores[category_name] = vs
    # نحتفظ بـ k=15 لضمان دقة السياق كما اتفقنا
    retrievers[category_name] = vs.as_retriever(search_kwargs={"k": 15})

@app.on_event("startup")
async def startup_event():
    for cat in CATEGORIES:
        initialize_category(cat)

# ==========================================
# 2. إعداد سلسلة المحادثة (المرنة والمنسقة)
# ==========================================
template = """أنت مستشار ذكي لمنصة "حي ابتكار" التابعة لجامعة الإمام عبدالرحمن بن فيصل.

قواعد اللغة الصارمة جداً (CRITICAL):
1. أجب بنفس لغة المستخدم تماماً. إذا كان السؤال بالإنجليزية يجب أن تكون الإجابة كاملة باللغة الإنجليزية (100% English). وإذا كان بالعربية أجب بالعربية.
2. يُمنع منعاً باتاً دمج اللغتين في نفس الجملة. قم بترجمة المعلومات الموجودة في السياق إلى لغة المستخدم قبل صياغة الرد.

قواعد المحتوى والتنسيق:
1. منصة "حي ابتكار" تضم 4 أقسام رئيسية دائماً هي: (براءات الاختراع والتقنيات، المشاريع البحثية والمراكز، الشركات الناشئة، الفرص الاستثمارية). لا تنسَ أي قسم عند الترحيب بالمستخدم.
2. عند إجابتك عن أي قسم، ابدأ إجابتك دائماً بهذه العبارة حرفياً مع وضع اسم القسم المناسب: "بناءً على المستجدات في منصة حي ابتكار، إليك أحدث [اسم القسم] لدينا:" (أو ترجمتها للإنجليزية إذا كان السؤال بالإنجليزية: "Based on the latest updates in the Hayy Ibtikar platform, here are our latest [Department Name]:").
3. اعتمد على <سياق_المعلومات> للإجابة عن التفاصيل.
4. استخدم تنسيقاً نظيفاً ومرتباً: استخدم النقاط (Bullet points) بشكل واضح، ولا تدمج الكلمات الإنجليزية داخل الجمل العربية لتجنب تشوه التنسيق.

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
مهمتك تحليل سؤال المستخدم وتحديد القسم الأنسب للإجابة عليه بدقة، اختر قسماً واحداً فقط من القائمة:
- patents 
- research 
- startups 
- investments 
- all 

سؤال المستخدم: {question}

أجب بكلمة واحدة فقط باللغة الإنجليزية من الكلمات المذكورة أعلاه.
القسم المختار:"""
router_prompt = PromptTemplate.from_template(router_template)
router_chain = router_prompt | llm | StrOutputParser()

# ==========================================
# 3. مسار الدردشة
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        category = router_chain.invoke({"question": request.message}).strip().lower()
        
        all_docs = []
        if category in retrievers:
            docs = retrievers[category].invoke(request.message)
            all_docs.extend(docs)
        else:
            for cat in retrievers:
                docs = retrievers[cat].invoke(request.message)
                all_docs.extend(docs)
                
        context_text = format_docs(all_docs[:15])
        
        response = qa_chain.invoke({
            "context": context_text,
            "history": request.history,
            "question": request.message
        })
        
        # إرسال القسم المختار مع الإجابة لعرضه في الواجهة
        return {"reply": response, "category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 4. مسار رفع البيانات (مع الحفظ التلقائي)
# ==========================================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Haider414/patents-chatbot")

@app.post("/upload-data")
async def upload_data(category: str = Form(...), file: UploadFile = File(...)):
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="قسم غير صالح.")
        
    try:
        content_bytes = await file.read()
        new_data = json.loads(content_bytes.decode("utf-8"))
        file_path = CATEGORIES[category]["file"]
        
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            
        combined_data = existing_data + (new_data if isinstance(new_data, list) else [new_data])
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
            
        initialize_category(category)
        
        # الرفع التلقائي إلى GitHub
        if GITHUB_TOKEN and GITHUB_REPO:
            try:
                url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{file_path}"
                headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                get_response = requests.get(url, headers=headers)
                sha = get_response.json().get("sha", "") if get_response.status_code == 200 else ""
                
                encoded_content = base64.b64encode(json.dumps(combined_data, ensure_ascii=False, indent=4).encode("utf-8")).decode("utf-8")
                put_data = {"message": f"Auto-update {category}", "content": encoded_content, "branch": "main"}
                if sha: put_data["sha"] = sha
                    
                requests.put(url, headers=headers, json=put_data)
            except Exception as e:
                print(f"GitHub Sync Error: {e}")

        return {"status": "success", "message": f"تم التحديث والحفظ بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))