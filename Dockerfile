
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Train the Rasa model (optional if pre-trained)
# RUN rasa train

# Run Rasa server
CMD ["rasa","run", "--enable-api", "--cors", "*"]
