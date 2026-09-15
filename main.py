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
1. إذا كان السؤال شخصياً أو مبنياً على حوار سابق، استخرج الإجابة من قسم <تاريخ_المحادثة>.
2. إذا كان السؤال عن الحي (براءات، أبحاث، شركات، استثمارات)، استخرج الإجابة من قسم <سياق_المعلومات>.
3. لا تقل "السياق لا يحتوي على معلومات" إذا كانت الإجابة موجودة في تاريخ المحادثة.

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
# 3. مسار الدردشة (الذي سنطوره للوكيل الموجه)
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # كحل مؤقت قبل برمجة الوكيل الموجه، سنقوم بالبحث في جميع الأقسام
        all_docs = []
        for cat in retrievers:
            docs = retrievers[cat].invoke(request.message)
            all_docs.extend(docs)
            
        context_text = format_docs(all_docs[:15])
        
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
        
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            
        combined_data = existing_data + (new_data if isinstance(new_data, list) else [new_data])
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
            
        initialize_category(category)

        # يمكننا لاحقاً تفعيل رفع الملفات لـ GitHub لكل قسم بنفس الطريقة

        return {"status": "success", "message": f"تم تحديث بيانات قسم {category} بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))