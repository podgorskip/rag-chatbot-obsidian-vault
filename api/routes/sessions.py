from fastapi import APIRouter, HTTPException, Depends

from api.dependencies import get_chatbot
from chatbot.chatbot import Chatbot

router = APIRouter()

@router.get("/sessions")
def list_sessions(chatbot: Chatbot = Depends(get_chatbot)):
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")
    return chatbot.list_sessions()

@router.post("/sessions")
def create_session(chatbot: Chatbot = Depends(get_chatbot)):
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")
    sid = chatbot.create_session()
    return {"session_id": sid}

@router.get("/sessions/{session_id}/history")
def get_history(
        session_id: str,
        chatbot: Chatbot = Depends(get_chatbot)
):
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")
    return chatbot.get_display_history(session_id)

@router.delete("/sessions/{session_id}")
def delete_session(
        session_id: str,
        chatbot: Chatbot = Depends(get_chatbot)
):
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")
    chatbot.delete_session(session_id)
    return {"status": "ok"}