{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4733137e-e185-4090-9b87-9c02295416cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "const io = require('socket.io')(3000, {\n",
    "  cors: {\n",
    "    origin: '*',\n",
    "  }\n",
    "});\n",
    "io.on('connection', (socket) => {\n",
    "  console.log('A user connected');\n",
    "\n",
    "io.on('connection', (socket) => {\n",
    "  console.log('A user connected');\n",
    "\n",
    "  socket.on('user_message', async (message) => {\n",
    "    console.log('Received user message:', message);\n",
    "\n",
    "    const response = await fetch('http://localhost:5005/webhooks/socketio/webhook', {\n",
    "      method: 'POST',\n",
    "      headers: {\n",
    "        'Content-Type': 'application/json',\n",
    "      },\n",
    "      body: JSON.stringify({\n",
    "        sender: socket.id,\n",
    "        message: message,\n",
    "      }),\n",
    "    });\n",
    "\n",
    "    const data = await response.json();\n",
    "    console.log('Rasa response:', data);\n",
    "\n",
    "    if (data && data.length > 0) {\n",
    "      const botMessage = data[0].text;\n",
    "      socket.emit('bot_message', botMessage);\n",
    "    } else {\n",
    "      socket.emit('bot_message', 'Sorry, I did not understand that.');\n",
    "    }\n",
    "  });\n",
    "\n",
    "  socket.on('disconnect', () => {\n",
    "    console.log('A user disconnected');\n",
    "  });\n",
    "});\n"
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
