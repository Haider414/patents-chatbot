import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

# جلب مفتاح API من متغيرات البيئة بأمان
API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Google API Key is not set in environment variables.")

app = FastAPI(title="Patents Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. تهيئة التضمينات (Embeddings) والنموذج (LLM) عبر Google لتوفير الذاكرة
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY, temperature=0.3)

# 2. بناء أو تحميل قاعدة بيانات المتجهات (Vector Store)
def initialize_vectorstore():
    index_path = "faiss_index"
    # إذا تم إنشاء القاعدة مسبقاً، سيتم تحميلها مباشرة
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # قراءة ملف براءات الاختراع وإنشاء القاعدة في حال عدم وجودها
        try:
            with open("patents.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # تحويل البيانات إلى مستندات (يفترض أن الملف عبارة عن قائمة كائنات JSON)
            documents = [Document(page_content=json.dumps(item, ensure_ascii=False)) for item in data]
            
            # إنشاء قاعدة FAISS جديدة
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(index_path)
            return vectorstore
        except Exception as e:
            print(f"Error loading patents.json: {e}")
            return FAISS.from_texts(["لا توجد بيانات حالياً."], embeddings)

vectorstore = initialize_vectorstore()

# ضبط الاسترجاع لجلب أفضل النتائج
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = qa_chain.invoke({"query": request.message})
        return {"reply": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))