import torch as tc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nlp_model.utils import get_device, to_device

class ModelAndTokenizer():
    """
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_tokenizer(self, prompt, padding="longest", truncation="longest_first"):
        """ """
        inputs = self.tokenizer(
            prompt,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        return to_device(inputs, self.device)
    
    def apply_tokenizer(self, dataset, padding="longest", truncation=True):
        data_tokenized = dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=truncation, padding=padding), 
            batched=True
        )
        return data_tokenized

    def decode_tokens(self, inputs, skip_special_tokens=True):
        """ """
        decode_text = []
        for seq in inputs:
            decode_text.append(self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens))
        return decode_text

    def load_classification_model(self, attentions=False):
        """ """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            output_attentions=attentions
        ).to(self.device)
        return model