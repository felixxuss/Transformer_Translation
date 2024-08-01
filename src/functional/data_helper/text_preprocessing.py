import re
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)

def remove_non_utf8(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

def remove_urls(text):
    return re.sub(r'http\S+|www.\S+', '', text)

def remove_html_tags(text):
    return re.sub('<.*?>', '', text)

def filter_characters(text):
    allowed_chars = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\|_+*¥"
    return ''.join(c for c in text if c in allowed_chars)

def lower_case(text):
    return text.lower()

def check_text_length(text, max_length=64, min_length=5):
    return len(text) <= max_length and len(text) >= min_length

def check_task_ratio(de_text, en_text, max_ratio=1.5):
    de_len = len(de_text)
    en_len = len(en_text)
    ratio = de_len / en_len
    return ratio <= max_ratio and ratio >= (1/max_ratio)

def clean_data(text: str) -> str:
    text = remove_non_utf8(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = filter_characters(text)
    text = lower_case(text)
    return text

def parse_text_list(text_data):
    """The function parses the text data and returns a list of dictionaries with the keys "de" and "en".

    Args:
        text_data (list[dict(str,str)]): _description_

    Returns:
        list[dict(str,str)]: Cleaned text data
    """
    result = []
    for text_dict in tqdm(text_data, desc="Parsing text data"):
        de_text = text_dict["de"]
        en_text = text_dict["en"]

        if not check_task_ratio(de_text, en_text):
            continue
        
        if not check_text_length(de_text) or not check_text_length(en_text):
            continue

        result.append({"de": clean_data(de_text), "en": clean_data(en_text)})
    logger.debug(f"Found {len(result)} ({round(len(result)/len(text_data)*100,2)}%) valid text pairs.")
    return result