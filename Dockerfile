FROM python:3.11-slim

WORKDIR /app

COPY /requirements.txt .

RUN apt-get clean && apt-get -y update && apt-get install -y build-essential cmake ffmpeg libsm6 libxext6 libopenblas-dev liblapack-dev libopenblas-dev liblapack-dev

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "--server.port=8501", "--server.enableCORS", "false", "main.py"]
