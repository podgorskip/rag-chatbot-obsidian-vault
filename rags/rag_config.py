class Config:
  def __init__(self):
    self.MAX_CONTEXT_TOKENS = 2000
    self.MIN_SIMILARITY = 0.5
    self.DELTA_CUTOFF = 0.08
    self.TOP_K = 0.5
    self.REPHRASE_PROMPT = ("You are a helpful assistant. Rephrase the following user query to be more descriptive and "
                            "search-engine friendly. Output ONLY the rephrased query.")
    self.ANSWER_PROMPT = (
        "You are a fast, sophisticated, direct assistant. "
        "Base only on the context attached. Use your knowledge only to understand query, not to generate answer. "
        "CRITICAL REQUIREMENTS: "
        "- Think fast. "
        "- Provide the COMPLETE answer for all parts of the user's request. "
        "- Be efficient with words but do not omit requested sections. "
        "- Do not use meta-talk like 'Answering your question' or 'Here is your itinerary'. "
        "- Use Markdown headers (##) for days, steps and bullet points for activities. "
        "- If you don't know the answer, say so and stop."
    )