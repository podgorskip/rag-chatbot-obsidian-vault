from pydantic import BaseModel

class MessageRequest(BaseModel):
    message: str
    session_id: str
    allow_external: bool | None = None

class MessageResponse(BaseModel):
    answer: str | None = None
    total_tokens: int = 0
    needs_confirmation: bool = False

class SettingsPayload(BaseModel):
    vault_path:      str
    exclude_folders: str
    knowledge_base:  str