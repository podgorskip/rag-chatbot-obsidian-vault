from fastapi import Request
from chatbot.chatbot import Chatbot

def get_chatbot(request: Request) -> Chatbot:
    chatbot = request.app.state.chatbot
    if chatbot is None:
        raise RuntimeError("Chatbot not initialized")
    return chatbot