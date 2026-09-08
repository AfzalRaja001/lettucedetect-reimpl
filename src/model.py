"""ModernBERT with a token-classification head for hallucination detection.

AutoModelForTokenClassification is a Hugging Face wrapper that takes any
encoder backbone (ModernBERT here) and attaches a linear layer on top,
mapping each token's hidden vector to `num_labels` class scores. It also
implements the loss computation itself: given `labels` at train time, it
computes cross-entropy per token and automatically skips any position whose
label is -100 - which is exactly the context/question masking data_prep.py
already baked into the cached labels. We don't have to write that masking
logic twice.
"""

from transformers import AutoModelForTokenClassification, PreTrainedModel

MODEL_NAME = "answerdotai/ModernBERT-base"

# Attaching human-readable names to the two classes is optional, but it means
# a saved checkpoint's config.json records what "0" and "1" mean, so loading
# it later (or in a different script, or a year from now) doesn't require
# remembering an unwritten convention.
ID2LABEL = {0: "SUPPORTED", 1: "HALLUCINATED"}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}


def build_model(model_name: str = MODEL_NAME) -> PreTrainedModel:
    """Load a ModernBERT backbone with a randomly-initialized 2-class head.

    The backbone's pretrained weights are loaded as-is; only the
    classification head on top is new and untrained, which is what
    fine-tuning on RAGTruth exists to fix.

    :param model_name: Hugging Face Hub id of the backbone to load
    :return: a model ready to be passed to a Trainer
    """
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
