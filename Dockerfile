FROM python:3.9-slim as builder
WORKDIR /app 
COPY . .
RUN pip install --no-cache-dir -r requirements.txt  
EXPOSE 5005 
CMD ["rasa", "run", "--enable-api", "--cors", "*", "--debug"] 
