import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class Config:
    MODEL_PATH: str = "swdq/modernbert-ja-310m-nsfw"
    LABEL_DICT: dict = field(default_factory=lambda: {"usual": 0, "aegi": 1, "chupa": 2})

    @property
    def reverse_label_dict(self) -> dict:
        return {v: k for k, v in self.LABEL_DICT.items()}

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

class TextClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH)
        self.model.to(self.device)

    def classify_text(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits.softmax(dim=-1)[0].cpu().numpy()
        label_dict = self.config.LABEL_DICT
        reverse_label_dict = self.config.reverse_label_dict
        prob_dict = {reverse_label_dict[i]: float(f"{probs[i]:.3f}") for i in range(len(probs))}
        predicted_class = int(probs.argmax())
        predicted_label = reverse_label_dict[predicted_class]

        return {"label": predicted_label, "probs": prob_dict}