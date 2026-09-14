import streamlit as st
import requests

st.set_page_config(page_title="مساعد براءات الاختراع", page_icon="🤖")
st.title("🤖 مساعد براءات الاختراع الذكي")

API_URL = "https://patents-chatbot-1-ufrw.onrender.com/chat"
UPLOAD_URL = "https://patents-chatbot-1-ufrw.onrender.com/upload-patent"

# شريط جانبي لرفع المستندات الحية
with st.sidebar:
    st.header("📂 إدارة المستندات")
    uploaded_file = st.file_uploader("ارفع ملف براءة اختراع (JSON)", type=["json"])
    if uploaded_file is not None:
        if st.button("رفع وتحديث قاعدة البيانات"):
            with st.spinner("جاري تحديث الفهرس على السحاب..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
                    response = requests.post(UPLOAD_URL, files=files)
                    if response.status_code == 200:
                        st.success("تم التحديث بنجاح! البوت جاهز للبحث في البراءة الجديدة.")
                    else:
                        st.error("فشل التحديث من الخادم.")
                except Exception as e:
                    st.error(f"خطأ في الاتصال: {e}")

# بقية كود العرض والمحادثة...
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اسألني عن الاختراعات..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("يبحث في المستندات..."):
        try:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
            response = requests.post(API_URL, json={"message": prompt, "history": history_text})
            response.raise_for_status()
            bot_reply = response.json().get("reply", "لم أتمكن من استخراج الإجابة.")
        except Exception as e:
            bot_reply = "عذراً، الخادم غير متاح حالياً."

    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})