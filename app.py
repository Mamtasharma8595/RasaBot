{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c1c73b44-5b20-401a-a70f-88a196f45be7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import requests\n",
    "\n",
    "# Set up the Streamlit page\n",
    "st.title(\"Chat with Rasa Bot\")\n",
    "\n",
    "# Input box for user messages\n",
    "user_message = st.text_input(\"You:\", \"\")\n",
    "\n",
    "# Button to send the message\n",
    "if st.button(\"Send\"):\n",
    "    if user_message:\n",
    "        # Send the user message to Rasa server\n",
    "        response = requests.post(\"http://localhost:5005/webhooks/rest/webhook\", json={\"sender\": \"user\", \"message\": user_message})\n",
    "        \n",
    "        # Get the response and display it\n",
    "        if response.status_code == 200:\n",
    "            rasa_response = response.json()\n",
    "            for r in rasa_response:\n",
    "                st.write(\"Bot:\", r.get(\"text\", \"Sorry, I did not understand that.\"))\n",
    "        else:\n",
    "            st.error(\"Error: Unable to connect to the Rasa server.\")\n",
    "    else:\n",
    "        st.warning(\"Please type a message to send.\")\n"
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
