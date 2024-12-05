# from flask import Flask, render_template
# from flask_socketio import SocketIO, emit
# import rasa
# import asyncio
# from rasa.core.agent import Agent

# app = Flask(__name__)
# socketio = SocketIO(app,async_mode='eventlet')
# loop = asyncio.get_event_loop()
# async def load_rasa_model():
#     model_path = "models/your_model.tar.gz"
#     return await loop.run_in_executor(None, Agent.load, model_path)

# # Declare the agent variable as None to be populated once loaded
# agent = None
# def load_agent():
#     global agent
#     agent = loop.run_until_complete(load_rasa_model())
    
# @app.route('/')
# def index():
#     return render_template('index.html')
    
# # Route for the home page
# @socketio.on('message')
# async def handle_message(message):
#     print(f"Received message: {message}")

#     if agent is None:
#         emit('response', {'message': "Sorry, the model is not loaded yet."})
#         return

#     # Pass the message to the Rasa agent and get a response
#     responses = await agent.handle_text(message)
#     print(f"Rasa response: {responses}")
    
#     # Emit the response to the client
#     emit('response', {'message': responses[0]['text']})

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=5000)