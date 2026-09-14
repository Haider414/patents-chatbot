# استخدام صورة بايثون الرسمية الخفيفة
FROM python:3.10-slim

# إعداد المجلد العملي داخل الحاوية
WORKDIR /app

# تثبيت المتطلبات الأساسية للنظام لتجنب مشاكل مكتبات التضمينات
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات أولاً للاستفادة من التخزين المؤقت (Cache)
COPY requirements.txt .

# تثبيت مكتبات بايثون
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع إلى الحاوية
COPY . .

# فتح المنافذ وتشغيل السيرفر باستخدام uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port $PORT