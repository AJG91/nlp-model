import torch as tc
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase, PreTrainedModel
)
from datasets import Dataset, DatasetDict
from typing import Union, Iterable
from nlp_model.utils import get_device, to_device

class ModelAndTokenizer():
    """
    A class for loading and managing a tokenizer and classification model.

    This class contains all the functionality necessary to load
    an encoding model, as well as functions for loading and applying
    a tokenizer with the option of using a data collator.
        
    Attributes
    ----------
    model_name : str
        Name of model that will be loaded.
    device : tc.device
        Device that tensors will be moved to.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer instance loaded from the pretrained model.

    Parameters
    ----------
    model_name : str
        Name of model that will be loaded.
    """
    def __init__(
        self, 
        model_name: str
    ):
        self.model_name = model_name
        self.device = get_device()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    def load_tokenizer(
        self, 
        prompt: str, 
        padding: Union[bool, str] = True, 
        truncation: Union[bool, str] = True
    ) -> dict[str, tc.Tensor]:
        """
        Tokenizes a text prompt (or list of prompts) and moves the 
        resulting tensors to the target device.

        Parameters
        ----------
        prompt : str or list[str]
            The input text or list of texts to tokenize.
        padding : bool or str, optional (default=True)
            Denotes the padding technique to use.
            If True, pad to the longest sequence in the batch.
            If False, does not pad.
        truncation : bool or str, optional (default=True)
            Denotes the truncation technique to use.
            If True, truncates to the model's maximum length.
            If False, does not truncate.
    
        Returns
        -------
        dict[str, tc.Tensor]
            A mapping from input field names (e.g. `'input_ids'`, `'attention_mask'`) 
            to PyTorch tensors located on `self.device`.
        """
        inputs = self.tokenizer(
            prompt,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        return to_device(inputs, self.device)
    
    def apply_tokenizer(
        self, 
        dataset: Union[Dataset, DatasetDict], 
        padding: Union[bool, str] = True, 
        truncation: Union[bool, str] = True
    ) -> Union[Dataset, DatasetDict]:
        """
        Tokenizes the text column of a Hugging Face dataset using the model's tokenizer.
        
        Parameters
        ----------
        dataset : datasets.Dataset or datasets.DatasetDict
            The dataset or dataset dictionary whose `"text"` column will be tokenized.
        padding : bool or str, optional (default=True)
            Denotes the padding technique to use.
            If True, pad to the longest sequence in the batch.
            If False, does not pad.
        truncation : bool or str, optional (default=True)
            Denotes the truncation technique to use.
            If True, truncates to the model's maximum length.
            If False, does not truncate.
    
        Returns
        -------
        datasets.Dataset or datasets.DatasetDict
            A new dataset with tokenized columns (e.g., `'input_ids'`, `'attention_mask'`) 
            and without the original `"text"` column.
        """
        data_tokenized = dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                padding=padding,
                truncation=truncation
            ), 
            batched=True,
            remove_columns=["text"]
        )
        return data_tokenized
    
    def data_collator(self) -> DataCollatorWithPadding:
        """
        Creates a data collator for dynamic padding during batching.

        Returns
        -------
        DataCollatorWithPadding
            A data collator that uses the current tokenizer to dynamically 
            pad input sequences to the length of the longest example in each batch.
        """
        return DataCollatorWithPadding(tokenizer=self.tokenizer)

    def decode_tokens(
        self, 
        inputs: Union[Iterable[list[int]], tc.Tensor], 
        skip_special_tokens: bool = True
    ) -> list[str]:
        """
        Decodes a batch of token ID sequences into text strings.
        
        Parameters
        ----------
        inputs : list[int] or tc.Tensor
            A collection of token ID sequences to decode. 
            Can be a list of lists of token IDs or a tensor of shape 
            `(batch_size, sequence_length)`.
        skip_special_tokens : bool, optional (default=True)
            Used to remove special tokens (e.g., `[CLS]`, `[PAD]`) from the decoded text.
            If True, skips special tokens.
            If False, includes them.
    
        Returns
        -------
        list of str
            A list of decoded text strings.
        """
        decode_text = []
        for sequence in inputs:
            decode_text.append(
                self.tokenizer.decode(
                    sequence, 
                    skip_special_tokens=skip_special_tokens
                )
            )
        return decode_text

    def load_classification_model(
        self, 
        attentions: bool = False
    ) -> PreTrainedModel:
        """
        Loads a pretrained sequence classification model and 
        moves it to the target device.
        
        Parameters
        ----------
        attentions : bool, optional (default=True)
            Enables attention outputs in the model.
    
        Returns
        -------
        Any
            A sequence classification model instance located on `self.device`.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            output_attentions=attentions
        ).to(self.device)
        return model