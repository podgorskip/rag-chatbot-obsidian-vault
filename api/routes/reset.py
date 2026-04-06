from fastapi import APIRouter, HTTPException, Depends

from api.dependencies import get_chatbot
from chatbot.chatbot import Chatbot

router = APIRouter()

@router.post("/reset")
def reset(
        payload: dict,
        chatbot: Chatbot = Depends(get_chatbot)
):
    if chatbot is None:
        raise HTTPException(503, "Chatbot not initialized.")
    sid = payload.get("session_id")
    if sid:
        chatbot.delete_session(sid)
    return {"status": "ok"}