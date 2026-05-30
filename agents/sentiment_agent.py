NEGATIVE_TERMS = {"angry", "bad", "complaint", "disappointed", "frustrated", "late", "unacceptable"}
POSITIVE_TERMS = {"great", "happy", "thanks", "thank", "resolved"}


def sentiment_agent(text):
    tokens = set(text.lower().replace(".", " ").replace(",", " ").split())

    if tokens & NEGATIVE_TERMS:
        return "negative"

    if tokens & POSITIVE_TERMS:
        return "positive"

    return "neutral"
