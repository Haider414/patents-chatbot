import os
import json
from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Patents Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY, temperature=0.3)

def initialize_vectorstore():
    index_path = "faiss_index"
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        try:
            with open("patents.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            documents = [Document(page_content=json.dumps(item, ensure_ascii=False)) for item in data]
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(index_path)
            return vectorstore
        except Exception as e:
            print(f"Error loading patents.json: {e}")
            return FAISS.from_texts(["لا توجد بيانات حالياً."], embeddings)

vectorstore = initialize_vectorstore()

# ضبط الاسترجاع مع تحديد 15 نتيجة لضمان دقة البحث
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# بناء سلسلة الاسترجاع الحديثة (LCEL) الموفرة للذاكرة
template = """استخدم السياق التالي للإجابة على السؤال.
السياق:
{context}

السؤال: {question}
الإجابة:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = qa_chain.invoke(request.message)
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))