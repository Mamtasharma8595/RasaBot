FROM python:3.9-slim 
WORKDIR /app 
COPY . .
RUN pip install --no-cache-dir -r requirements.txt  
EXPOSE 5005 
ENTRYPOINT ["sh", "-c", "rasa run --enable-api --cors '*' --debug -p ${PORT:-5005}"]