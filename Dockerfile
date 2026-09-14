FROM python:3.10-slim
WORKDIR /app

# Copy requirements from app/ requirements.txt and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY app/code/ ./code/
COPY saved_models/ ./saved_models/

WORKDIR /app/code
EXPOSE 8050
CMD ["python", "app.py"]