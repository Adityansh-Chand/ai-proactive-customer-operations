def sentiment_agent(text):
    if "complaint" in text.lower():
        return "negative"
    return "neutral"
