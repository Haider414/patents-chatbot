import streamlit as st
import requests
import base64 # تأكد من إضافة هذا في أعلى الملف إذا لم يكن موجوداً

# تأكد من أن هذا هو رابط الخادم الخاص بك على Render
API_URL = "https://patents-chatbot-1-ufrw.onrender.com" 

# إعداد الصفحة لتعكس هوية حي ابتكار
st.set_page_config(page_title="حي ابتكار - البوابة الرقمية", page_icon="💡", layout="centered")

st.title("💡 البوابة الرقمية لحي ابتكار")
st.write("مرحباً بك في المساعد الذكي لجامعة الإمام عبدالرحمن بن فيصل. يمكنك سؤالي عن براءات الاختراع، الأبحاث، الشركات الناشئة، أو الفرص الاستثمارية.")

# ==========================================
# القائمة الجانبية: لوحة تحكم إدارة المستندات
# ==========================================
with st.sidebar:
    st.header("📂 إدارة مستندات حي ابتكار")
    
    # القاموس لربط أسماء الأقسام بالعربية مع المفاتيح الإنجليزية
    category_options = {
        "براءات الاختراع والتقنيات": "patents",
        "المشاريع البحثية والمراكز": "research",
        "الشركات الناشئة": "startups",
        "الفرص الاستثمارية": "investments"
    }
    
    # قائمة منسدلة لاختيار القسم
    selected_display = st.selectbox("اختر القسم الذي تريد تحديثه:", list(category_options.keys()))
    selected_category = category_options[selected_display]
    
    # رفع الملف
    uploaded_file = st.file_uploader(f"ارفع ملف (JSON) الخاص بـ {selected_display}", type=["json"])
    
    if st.button("رفع وتحديث قاعدة البيانات"):
        if uploaded_file is not None:
            with st.spinner('جاري التحديث وبناء الفهرس المتجهي...'):
                # نرسل الملف كـ File والقسم كـ Form Data
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
                data = {"category": selected_category}
                
                try:
                    response = requests.post(f"{API_URL}/upload-data", files=files, data=data)
                    if response.status_code == 200:
                        st.success(f"تم تحديث بيانات قسم '{selected_display}' بنجاح!")
                    else:
                        st.error(f"فشل التحديث! تفاصيل الخطأ: {response.text}")
                except Exception as e:
                    st.error(f"حدث خطأ في الاتصال بالخادم: {e}")
        else:
            st.warning("الرجاء اختيار ملف أولاً.")

# ==========================================
# واجهة الدردشة الذكية
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اسألني عن أي شيء في حي ابتكار..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # تجهيز سجل المحادثة لدعم الذاكرة
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]])

    with st.chat_message("assistant"):
        with st.spinner('جاري البحث في ملفات الحي...'):
            try:
                payload = {"message": prompt, "history": history_text}
                res = requests.post(f"{API_URL}/chat", json=payload)
                
                if res.status_code == 200:
                    response_data = res.json()
                    reply = response_data.get("reply", "عذراً، لم أتمكن من صياغة الإجابة.")
                    category = response_data.get("category", "all")
                    
                    # 1. الإصلاح الأهم: حفظ الإجابة في الذاكرة فوراً قبل أي شيء آخر!
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    
                    # ترجمة اسم القسم لعرضه بشكل جميل
                    category_names = {
                        "patents": "براءات الاختراع والتقنيات 💡",
                        "research": "المشاريع البحثية والمراكز 🔬",
                        "startups": "الشركات الناشئة 🚀",
                        "investments": "الفرص الاستثمارية 📈",
                        "all": "بحث شامل 🌐"
                    }
                    display_cat = category_names.get(category, category)
                    
                    # عرض إشعار ذكاء الوكيل
                    st.info(f"🤖 **الوكيل الموجه:** تم تحليل سؤالك وتوجيهه إلى قسم [{display_cat}]")
                    
                    # عرض الإجابة بتنسيق يدعم اللغتين والاتجاهين بشكل سليم
                    st.markdown(f'<div dir="auto" style="text-align: justify;">{reply}</div>', unsafe_allow_html=True)
                    
                    # 2. تحديد مفتاح الصوت بناءً على طول الرسائل (ناقص 1 لأننا أضفنا الرسالة للتو)
                    audio_key = f"audio_{len(st.session_state.messages) - 1}"
                    
                    # إذا كان الصوت قد تم توليده وحفظه مسبقاً، اعرض مشغل الصوت مباشرة
                    if audio_key in st.session_state:
                        st.audio(st.session_state[audio_key], format="audio/mp3")
                    else:
                        # إذا لم يكن موجوداً، اعرض زر الاستماع
                        if st.button("🎙️ استمع للإجابة", key=f"voice_btn_{len(st.session_state.messages) - 1}"):
                            with st.spinner("جاري توليد الصوت الاحترافي..."):
                                try:
                                    # 3. استخدام المتغير الديناميكي API_URL بدلاً من localhost
                                    speech_res = requests.post(
                                        f"{API_URL}/speak", 
                                        json={"text": reply}
                                    )
                                    if speech_res.status_code == 200:
                                        audio_b64 = speech_res.json().get("audio_base64")
                                        # حفظ الصوت في الذاكرة حتى لا يختفي بعد الضغط
                                        st.session_state[audio_key] = base64.b64decode(audio_b64)
                                        # إعادة تحديث الواجهة فوراً لإظهار مشغل الصوت
                                        st.rerun() 
                                    else:
                                        st.error("عذراً، حدث خطأ أثناء الاتصال بمحرك الصوت.")
                                except Exception as e:
                                    st.error(f"فشل الاتصال بالخادم: {e}")
                else:
                    st.error("حدث خطأ أثناء جلب الإجابة من الخادم.")
            except Exception as e:
                st.error(f"فشل الاتصال بالخادم. تأكد من أن سيرفر Render يعمل. التفاصيل: {e}")