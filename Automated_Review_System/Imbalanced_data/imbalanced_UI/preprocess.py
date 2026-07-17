import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text):

    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess(text):

    text = clean_text(text)

    doc = nlp(text)

    words = []

    for token in doc:

        if not token.is_stop:
            words.append(token.lemma_)

    return " ".join(words)