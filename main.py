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

# تم التحديث هنا للنماذج الصحيحة لعام 2026 بناءً على توجيهك
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview", google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=API_KEY)

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

# تم التصحيح بناءً على طلبك السابق بخصوص المتغيرات
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

template = """أنت مساعد ذكي. اتبع هذه القواعد بصرامة:
1. إذا كان السؤال شخصياً أو مبنياً على حوار سابق (مثل "ما اسمي؟")، استخرج الإجابة من قسم <تاريخ_المحادثة>.
2. إذا كان السؤال عن الاختراعات، استخرج الإجابة من قسم <سياق_البراءات>.
3. لا تقل "السياق لا يحتوي على معلومات" إذا كانت الإجابة موجودة في تاريخ المحادثة.

<تاريخ_المحادثة>
{history}
</تاريخ_المحادثة>

<سياق_البراءات>
{context}
</سياق_البراءات>

السؤال الحالي: {question}
الإجابة:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# تعديل السلسلة لتقبل متغير history
qa_chain = prompt | llm | StrOutputParser()

class ChatRequest(BaseModel):
    message: str
    history: str = "" # إضافة متغير السجل

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # استرجاع المستندات بناءً على السؤال فقط
        docs = retriever.invoke(request.message)
        context_text = format_docs(docs)
        
        # تمرير السجل، السياق، والسؤال للنموذج
        response = qa_chain.invoke({
            "context": context_text,
            "history": request.history,
            "question": request.message
        })
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))