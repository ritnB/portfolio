# models/sentiment_model.py

from transformers import AutoModelForSequenceClassification

def load_sentiment_model(model_path: str):
    """
    Load a pretrained sentiment classification model from the given path.

    Args:
        model_path (str): Path to the local or remote model directory.
    Returns:
        transformers.PreTrainedModel: Loaded model instance.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_path)
