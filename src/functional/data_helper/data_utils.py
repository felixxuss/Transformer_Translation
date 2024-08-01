from torch.utils.data import Dataset, DataLoader
import torch

from datasets import load_dataset
import logging
import random
import json

from src.transformer.utils.tokenizer import get_tokenizers, check_tokenizer_files
from src.functional.data_helper.text_preprocessing import parse_text_list

logger = logging.getLogger(__name__)

def get_dataloader(mode="train", subset_n_batches=None, config=None):
    """The mode defines on which dataset the tokenizers are trained"""

    translation_dataset = get_translation_dataset(mode, subset_n_batches, config)
    
    shuffle_dataloader = True if mode == "train" else False
    
    translation_dataloader = DataLoader(translation_dataset, batch_size=config["BATCH_SIZE"], shuffle=shuffle_dataloader, drop_last=True)

    logger.debug(f"{mode}_data_loader: batch size: {config['BATCH_SIZE']} - total batches: {len(translation_dataloader)}")

    return translation_dataloader

def get_translation_dataset(mode, subset_n_batches=None, config=None):
    # get translation data
    text = load_translation_data(mode)
    
    # only choose n batches for faster testing
    if subset_n_batches:
        subset_size = config["BATCH_SIZE"] * subset_n_batches
        if config["RANDOM_BATCHES"]:
            text = random.sample(text, subset_size)
        else:
            text = text[:subset_size]
        logger.debug(f"{subset_size} sentences are used as {mode} data")

    # get tokenizers
    check_tokenizer_files() # check if needed tokenizer files have been created before
    en_tokenizer, de_tokenizer = get_tokenizers()

    translation_dataset = TranslationDataset(text, de_tokenizer, en_tokenizer, max_length=config["MAX_LEN"])

    return translation_dataset


class TranslationDataset(Dataset):
    def __init__(self, text, de_tokenizer, en_tokenizer, max_length=64):
        """The dataset class for the translation task.

        Args:
            text (list): A list of dictionaries with the keys "de" and "en".
            de_tokenizer (_type_): German tokenizer
            en_tokenizer (_type_): English tokenizer
            max_length (int, optional): Max sequence length. Defaults to 64.

        Returns:
            german_tokens: Tensor of shape (batch_size, max_length)
            english_tokens: Tensor of shape (batch_size, max_length)
            german_attention_mask: Tensor of shape (batch_size, max_length)
            english_attention_mask: Tensor of shape (batch_size, max_length)
        """
        self.text = text
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer
        self.pad_token_id_de = de_tokenizer.tokenizer.pad_token_id
        self.pad_token_id_en = en_tokenizer.tokenizer.pad_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        de_sentence, en_sentence = self.text[idx]["de"], self.text[idx]["en"]

        # Tokenize the sentences
        de_tokens = self.de_tokenizer.encode(de_sentence)
        en_tokens = self.en_tokenizer.encode(en_sentence)

        if len(de_tokens) > self.max_length or len(en_tokens) > self.max_length:
            logger.critical(f"Found a sentence that is longer than the max_length: de_len:{len(de_tokens)} | en_len:{len(en_tokens)}")

        # pad with BOS and EOS tokens
        de_tokens = [self.de_tokenizer.tokenizer.bos_token_id] + de_tokens + [self.de_tokenizer.tokenizer.eos_token_id]
        en_tokens = [self.en_tokenizer.tokenizer.bos_token_id] + en_tokens + [self.en_tokenizer.tokenizer.eos_token_id]

        # Pad or truncate the sequences to the specified max_length
        de_tokens = de_tokens + [self.pad_token_id_de] * (self.max_length - len(de_tokens))
        en_tokens = en_tokens + [self.pad_token_id_en] * (self.max_length - len(en_tokens))

        # calculate the attention mask
        # 1 where de_tokens != pad_token_id_de, 0 otherwise
        de_attention_mask = [1 if token != self.pad_token_id_de else 0 for token in de_tokens]
        en_attention_mask = [1 if token != self.pad_token_id_en else 0 for token in en_tokens]

        # assert that all shapes are equal
        assert len(de_tokens) == len(en_tokens) == len(de_attention_mask) == len(en_attention_mask) == self.max_length, f"Shape mismatch: de_tokens: {len(de_tokens)}, en_tokens: {len(en_tokens)}, de_attention_mask: {len(de_attention_mask)}, en_attention_mask: {len(en_attention_mask)}, max_length: {self.max_length}"

        return torch.tensor(de_tokens), torch.tensor(de_attention_mask), torch.tensor(en_tokens), torch.tensor(en_attention_mask)
    
def load_translation_data(text_type="train", cached=True):
    if text_type == "train":
        if cached:
            logger.debug(f"Using precalculated data from /data/parsed_text/")
            with open("data/parsed_text/parsed_train_text.json", "r") as f:
                parsed_text = json.load(f)
        else:
            logger.debug("Calculating the training data from scratch")
            dataset = load_dataset("wmt17", "de-en")
            train_data = dataset["train"]["translation"]
            parsed_text = parse_text_list(train_data)
    elif text_type == "test":
        if cached:
            logger.debug(f"Using precalculated data from /data/parsed_text/")
            with open("data/parsed_text/parsed_test_text.json", "r") as f:
                parsed_text = json.load(f)
        else:
            logger.debug("Calculating the test data from scratch")
            dataset = load_dataset("wmt17", "de-en")
            test_data = dataset["test"]["translation"]
            parsed_text = parse_text_list(test_data)
    elif text_type == "val":
        if cached:
            logger.debug(f"Using precalculated data from /data/parsed_text/")
            with open("data/parsed_text/parsed_val_text.json", "r") as f:
                parsed_text = json.load(f)
        else:
            logger.debug("Calculating the validation data from scratch")
            dataset = load_dataset("wmt17", "de-en")
            val_data = dataset["validation"]["translation"]
            parsed_text = parse_text_list(val_data)
    else:
        raise ValueError(f"Invalid text_type: {text_type}")

    return parsed_text

     

    