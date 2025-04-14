FROM python:3.11-slim

WORKDIR /app

COPY /requirements.txt .

RUN apt-get clean && apt-get -y update && apt-get install -y build-essential cmake ffmpeg libsm6 libxext6 libopenblas-dev liblapack-dev libopenblas-dev liblapack-dev

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "main.py"]
