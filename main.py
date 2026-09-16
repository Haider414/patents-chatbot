import os
import json
import base64
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
# سنستخدم requests للاتصال بـ ElevenLabs
from dotenv import load_dotenv
load_dotenv()

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
template = """You are a smart consultant for the "Hayy Ibtikar" platform at Imam Abdulrahman Bin Faisal University.
أنت مستشار ذكي لمنصة "حي ابتكار" التابعة لجامعة الإمام عبدالرحمن بن فيصل.

CRITICAL RULES (قواعد صارمة جداً):
1. Language Matching: 
   - If the user asks in English, you MUST answer entirely in English.
   - إذا سأل المستخدم بالعربية، أجب بالعربية بأسلوب احترافي.
2. Technical Terms: You MUST keep technical terms, AI model names (like YOLOv8, Mask R-CNN, Depth Anything), and company names in English. Do NOT translate or transliterate them to Arabic letters.
3. Formatting & Structure: 
   - Never use long narrative paragraphs. 
   - Always structure the details using clear bullet points (e.g., • اسم المشروع: ... • الحالة: ... • التفاصيل: ...).
4. Conversational Style: 
   - Do not repeat greetings if there is a chat history. Enter the subject directly.
5. Follow-up Questions (الأسئلة التفاعلية في النهاية): 
   يجب أن تنهي إجابتك دائماً بسؤالين محددين (مترجمين للغة المستخدم):
   - الأول: اسأل عما إذا كان يريد تفاصيل إضافية عن الموضوع الحالي (مثال: "هل ترغب في معرفة المزيد من التفاصيل حول هذا البحث؟").
   - الثاني: اسأل عما إذا كان يود استكشاف أقسام أخرى (مثال: "أو هل تود استكشاف بقية أقسام المنصة مثل براءات الاختراع أو الشركات الناشئة؟").

<تاريخ_المحادثة>
{history}
</تاريخ_المحادثة>

<سياق_المعلومات>
{context}
</سياق_المعلومات>

User Question / سؤال المستخدم: {question}
Response / الإجابة:"""

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

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

class SpeechRequest(BaseModel):
    text: str

@app.post("/speak")
async def generate_speech(request: SpeechRequest):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API Key is missing.")
        
    try:
        cleaned_text = clean_text_for_speech(request.text)
        
        # إعدادات ElevenLabs (يمكنك تغيير ID الصوت لاحقاً بصوت تفضله)
        # صوت Adam الافتراضي المجاني المدعوم للـ API
        voice_id = "pNInz6obpgDQGcFmaJgB" # Adam
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
            "text": cleaned_text,
            "model_id": "eleven_multilingual_v2", # يدعم العربية والإنجليزية بطلاقة
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code != 200:
            # اطبع الخطأ الفعلي القادم من ElevenLabs في التيرمنال لنراه بوضوح
            print(f"🔴 ElevenLabs API Error: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error: {response.text}")
            
        # تحويل الصوت إلى Base64 لإرساله بسهولة للواجهة
        audio_base64 = base64.b64encode(response.content).decode('utf-8')
        return {"audio_base64": audio_base64}
        
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

# ==========================================
# وظائف معالجة الصوت (Text-to-Speech)
# ==========================================
def clean_text_for_speech(text: str) -> str:
    # 1. إزالة علامات Markdown (النجمات، الشُرط، الروابط)
    text = re.sub(r'[*_#`\[\]()]', '', text)
    text = re.sub(r'-', ' ', text)
    
    # 2. إزالة التشكيل العربي (حركات الفتحة، الضمة، الكسرة، إلخ)
    arabic_diacritics = re.compile(r'[\u064B-\u065F]')
    text = arabic_diacritics.sub('', text)
    
    # 3. إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    return text