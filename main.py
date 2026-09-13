from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import requests

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. مفتاح جوجل الخاص بك
# قراءة المفتاح بأمان من متغيرات البيئة
API_KEY = os.environ.get("GOOGLE_API_KEY")

# 2. قراءة البيانات
with open("patents.json", "r", encoding="utf-8") as file:
    patents_data = json.load(file)

documents = []
for patent in patents_data:
    content = f"رقم الاختراع: {patent['id']}\nعنوان الاختراع: {patent['title']}\nالوصف: {patent['description']}"
    documents.append(Document(page_content=content))

# 3. التضمين المحلي
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# مصفوفة لحفظ سجل المحادثة مؤقتاً في الذاكرة
chat_history = []

# --- إعداد واجهة الـ API ---
app = FastAPI(title="Patents Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    user_message: str

@app.post("/chat")
async def chat_endpoint(request: MessageRequest):
    incoming_text = request.user_message
    
    # أ. استرجاع المعلومات بناءً على السؤال الحالي
    relevant_docs = retriever.invoke(incoming_text)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
    # ب. بناء الرسالة الموجهة للنموذج مع دمج السياق
    system_instruction = (
        "أنت مساعد ذكي خاص بجامعة للإجابة على استفسارات براءات الاختراع. "
        "استخدم المعلومات المسترجعة للإجابة بدقة، وإذا لم تكن الإجابة موجودة فاعتذر بلباقة."
    )
    
    current_prompt = (
        f"{system_instruction}\n\n"
        f"المعلومات المسترجعة من النظام:\n{context}\n\n"
        f"سؤال المستخدم الجديد: {incoming_text}"
    )
    
    # ج. تجهيز سجل المحادثة الكامل لإرساله لـ Gemini
    contents = []
    for turn in chat_history:
        contents.append(turn)
        
    # إضافة الرسالة الحالية إلى السجل المرسل
    contents.append({"role": "user", "parts": [{"text": current_prompt}]})
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": contents}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        
        ai_reply = response_json['candidates'][0]['content']['parts'][0]['text']
        
        # د. حفظ دورة الحوار في الذاكرة لاستخدامها لاحقاً
        chat_history.append({"role": "user", "parts": [{"text": incoming_text}]})
        chat_history.append({"role": "model", "parts": [{"text": ai_reply}]})
        
        return {"response": ai_reply}
        
    except Exception as e:
        print("خطأ في الاتصال:", response.text)
        return {"response": "عذراً، حدث خطأ في معالجة الذاكرة السياقية."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)