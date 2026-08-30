"""Sentiment classification.

Was a 12-word set membership test. Now the fitted classifier from
`training/train.py`; see `agents/classifiers.py`.
"""
from agents.classifiers import classify_sentiment


def sentiment_agent(text):
    label, _ = classify_sentiment(text)
    return label
