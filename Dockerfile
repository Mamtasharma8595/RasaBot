{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a6cff09-b2bf-40bb-843e-5431ccdb003c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Base image for Rasa\n",
    "FROM rasa/rasa:latest\n",
    "\n",
    "# Copy project files\n",
    "COPY . /app\n",
    "\n",
    "# Change working directory\n",
    "WORKDIR /app\n",
    "\n",
    "# Expose Rasa API port\n",
    "EXPOSE 5005\n",
    "\n",
    "# Train the Rasa model (optional if pre-trained)\n",
    "#RUN rasa train\n",
    "\n",
    "# Run Rasa server\n",
    "CMD [\"run\", \"--enable-api\", \"--cors\", \"*\"]\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}