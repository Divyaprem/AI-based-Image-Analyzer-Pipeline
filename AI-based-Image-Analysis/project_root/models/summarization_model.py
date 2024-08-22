from transformers import pipeline

def load_summarization_model():
    summarizer = pipeline("summarization")
    return summarizer

def generate_summary(summarizer, texts):
    summaries = [summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for text in texts]
    return summaries
