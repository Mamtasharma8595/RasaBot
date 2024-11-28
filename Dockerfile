
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
# Train the Rasa model (optional if pre-trained)
# RUN rasa train
EXPOSE 5005

# Run Rasa server
CMD ["rasa", "run", "--enable-api", "--cors", "*", "--debug"]
