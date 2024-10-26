FROM python:3.9-slim as builder
WORKDIR /app 
COPY requirements.txt .
COPY . .
RUN pip install --no-cache-dir -r requirements.txt  
EXPOSE 5005 
HEALTHCHECK CMD curl --fail http://localhost:5005 || exit 1
CMD ["rasa", "run", "--enable-api", "--cors", "*", "--debug"] 
