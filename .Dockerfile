FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both code and saved_models
COPY code/ ./code/
COPY saved_models/ ./saved_models/

WORKDIR /app/code
EXPOSE 8050
CMD ["python", "app.py"]