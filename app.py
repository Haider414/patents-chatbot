import streamlit as st
import requests

# إعدادات واجهة الصفحة
st.set_page_config(page_title="مساعد براءات الاختراع", page_icon="🤖")
st.title("🤖 مساعد براءات الاختراع الذكي")

# رابط الـ API الخاص بك على منصة Render
API_URL = "https://patents-chatbot-1-ufrw.onrender.com/chat"

# تهيئة ذاكرة الجلسة للاحتفاظ بالرسائل المعروضة على الشاشة
if "messages" not in st.session_state:
    st.session_state.messages = []

# رسم الرسائل السابقة على الشاشة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال سؤال المستخدم من مربع النص
if prompt := st.chat_input("اسألني عن الاختراعات (مثل: نظام الري أو الروبوت الطبي)..."):
    
    # عرض سؤال المستخدم فوراً في الواجهة
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # إرسال الطلب إلى السيرفر الخاص بك واستقبال الرد
    with st.spinner("يبحث في المستندات..."):
        try:
            # تجميع الرسائل السابقة كنص واحد (باستثناء السؤال الأخير لتجنب التكرار)
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
            
            # إرسال السؤال مع سجل المحادثة
            response = requests.post(API_URL, json={"message": prompt, "history": history_text})
            response.raise_for_status()
            bot_reply = response.json().get("reply", "لم أتمكن من استخراج الإجابة.")
        except Exception as e:
            bot_reply = "عذراً، الخادم السحابي غير متاح حالياً أو في وضع السبات."

    # عرض إجابة البوت في الواجهة
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})