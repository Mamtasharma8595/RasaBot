FROM python:3.8-slim 
WORKDIR /app 
COPY . . 
RUN pip install --no-cache-dir -r requirements.txt  
EXPOSE 5005 
CMD ["rasa", "run", "--enable-api", "--cors", "*", "--debug"] 
