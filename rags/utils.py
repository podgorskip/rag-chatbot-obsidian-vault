def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."

    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)

def estimate_tokens(text: str) -> int:
    return len(text) // 4