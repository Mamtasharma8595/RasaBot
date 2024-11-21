
# Base image for Rasa
FROM rasa/rasa:latest

# Copy project files
COPY . /app

# Change working directory
WORKDIR /app

# Expose Rasa API port
EXPOSE 5005

# Train the Rasa model (optional if pre-trained)
# RUN rasa train

# Run Rasa server
CMD ["run", "--enable-api", "--cors", "*"]
