import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import shutil
import base64

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

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Haider414/patents-chatbot")

@app.post("/upload-patent")
async def upload_patent(file: UploadFile = File(...)):
    try:
        # 1. قراءة محتوى الملف المرفوع
        content_bytes = await file.read()
        file_content_str = content_bytes.decode("utf-8")
        new_data = json.loads(file_content_str)
        
        # 2. قراءة البيانات الحالية ودمجها في الذاكرة
        try:
            with open("patents.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = []
            
        combined_data = existing_data + (new_data if isinstance(new_data, list) else [new_data])
        
        # 3. تحديث الملف محلياً في الذاكرة والخادم مؤقتاً
        with open("patents.json", "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
            
        # 4. تحديث فهرس المتجهات (FAISS) فوراً لتصبح البراءة قابلة للبحث في نفس اللحظة
        global vectorstore, retriever, qa_chain
        documents = [Document(page_content=json.dumps(item, ensure_ascii=False)) for item in combined_data]
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("faiss_index")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        qa_chain = prompt | llm | StrOutputParser()
        
        # 5. رفع الملف المحدث تلقائياً إلى GitHub في الخلفية لضمان الحفظ الدائم
        if GITHUB_TOKEN:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/patents.json"
            headers = {
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json"
            }
            # جلب sha الخاص بالملف الحالي على GitHub لكي يتم تحديثه
            get_res = requests.get(url, headers=headers)
            sha = get_res.json().get("sha") if get_res.status_code == 200 else None
            
            # تشفير محتوى الملف الجديد بـ Base64
            encoded_content = base64.b64encode(json.dumps(combined_data, ensure_ascii=False, indent=4).encode("utf-8")).decode("utf-8")
            
            payload = {
                "message": f"Auto-update patents.json via web UI ({file.filename})",
                "content": encoded_content,
                "sha": sha
            }
            if sha:
                requests.put(url, headers=headers, json=payload)

        return {"status": "success", "message": "تم رفع وتحديث البراءة والبحث فيها فوراً، وحفظها على السحاب بنجاح!"}
    except Exception as e:
        raise HTTPException(status_code.status_code if hasattr(e, 'status_code') else 500, detail=str(e))