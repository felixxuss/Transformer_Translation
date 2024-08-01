from tokenizers import Tokenizer, trainers, models, pre_tokenizers
from transformers import GPT2Tokenizer
from os import path
import json
import logging

logger = logging.getLogger(__name__)

class MyGPT2Tokenizer:
    def __init__(
            self,
            prefix="de",
            add_bos_token=False):
        
        self.tokenizer = GPT2Tokenizer(
            vocab_file=f"data/tokenizer/{prefix}_vocab.json", 
            merges_file=f"data/tokenizer/{prefix}_merges.json",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            add_bos_token=add_bos_token)
        
        self.special_tokens = {
            "BOS": self.tokenizer.bos_token_id,
            "EOS": self.tokenizer.eos_token_id,
            "PAD": self.tokenizer.pad_token_id
        }
        
    def encode(self, in_text):
        if isinstance(in_text, list):
            return [self.tokenizer.encode(text.lower()) for text in in_text]
        elif isinstance(in_text, str):
            return self.tokenizer.encode(in_text.lower())
        else:
            raise TypeError("Input must be of type list or string.")
    
    def decode(self, in_tokens):
        if isinstance(in_tokens[0], list):
            return [self.tokenizer.decode(tokens) for tokens in in_tokens]
        elif isinstance(in_tokens[0], int):
            return self.tokenizer.decode(in_tokens)
        else:
            raise TypeError("Input must be of type list or int.")

def get_tokenizers():
    """Returns the tokenizers for the translation task

    Returns:
        _type_: Returns first the english tokenizer, then the german tokenizer
    """
    return MyGPT2Tokenizer("en"), MyGPT2Tokenizer("de")

def check_tokenizer_files():
    if path.exists("data/tokenizer/de_vocab.json") or path.exists("data/tokenizer/de_merges.json") or path.exists("data/tokenizer/en_vocab.json") or path.exists("data/tokenizer/en_merges.json"):
        return
    else:
        # please use prepare_data_for_gpt_tokenizer to create the tokenizer files with the training data
        raise FileNotFoundError("The tokenizer files are not found. Please make sure that the files are in the correct location.")
    

"""
The code below this point is not needed anymore. It is only used once.
It can be used to create the vocab and merges file for the MyGPT2Tokenizer.
"""

def prepare_data_for_gpt_tokenizer(text_data):
    """Tokenizes the text data and saves the vocab and merges files for the GPT2Tokenizer.

    Args:
        text_data (list): List of dictionaries with the keys "de" and "en".
    """
    # check if the files are already created
    if path.exists("data/tokenizer/de_vocab.json") or path.exists("data/tokenizer/de_merges.json") or path.exists("data/tokenizer/en_vocab.json") or path.exists("data/tokenizer/en_merges.json"):
        return
    else:
        logger.debug("Tokenizers will be created. This may take a while.")

    de_tokenizer, en_tokenizer = tokenize_translation_task(text_data)

    # save complete tokenizers as json files
    de_tokenizer.save_tokenizer("de")
    en_tokenizer.save_tokenizer("en")

    # read the tokenizers and save the english vocab and merges files
    with open("data/tokenizer/en_tokenizer.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("data/tokenizer/en_vocab.json", "w") as f:
        json.dump(data["model"]["vocab"], f)
    with open("data/tokenizer/en_merges.json", "w") as f:
        json.dump(data["model"]["merges"], f)

    # read the tokenizers and save the german vocab and merges files
    with open("data/tokenizer/de_tokenizer.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    with open("data/tokenizer/de_vocab.json", "w") as f:
        json.dump(data["model"]["vocab"], f)
    with open("data/tokenizer/de_merges.json", "w") as f:
        json.dump(data["model"]["merges"], f)

def tokenize_translation_task(text_data, max_vocab_length=50_000):
    """Tokenizes the text data and returns the vocabularies and tokenizers.

    Args:
        text_data (list): List of dictionaries with the keys "de" and "en".
        max_vocab_length (int, optional): Number of maximal vocabs. Defaults to 50_000.

    Returns: German and English Vocabulary, German and English Tokenizer
    """
    # split german and english text
    de_data = [element["de"] for element in text_data]
    en_data = [element["en"] for element in text_data]

    # create seperate tokenizers
    de_tokenizer = HuggBPETokenizer(max_vocab_size=max_vocab_length)
    en_tokenizer = HuggBPETokenizer(max_vocab_size=max_vocab_length)

    # train tokenizers
    de_tokenizer.train_from_iterator(de_data)
    en_tokenizer.train_from_iterator(en_data)

    return de_tokenizer, en_tokenizer


class HuggBPETokenizer:
    def __init__(self, max_vocab_size=50_000):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        self.trainer = trainers.BpeTrainer(
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
            unk_token="[UNK]",
            vocab_size=max_vocab_size,            
        )

    def train_from_iterator(self, iterator):
        self.tokenizer.train_from_iterator(iterator, trainer=self.trainer)

    def save_tokenizer(self, prefix: str):
        self.tokenizer.save(f"data/tokenizer/{prefix}_tokenizer.json", pretty=True)

    def load_tokenizer(self, prefix: str):
        self.tokenizer = Tokenizer.from_file(f"data/tokenizer/{prefix}_tokenizer.json")

    def encode(self, text):
        return self.tokenizer.encode(text).ids
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens)