import os
import logging

# modules
from src.transformer.utils.model_utils import load_pretrained_model
from src.transformer.utils.tokenizer import get_tokenizers
from src.functional.sampler_helper.sample_utils import greedy_translate
import src.functional.utils as utils

def sample(run_name):
    """Method to calculate a translation for a english sentence given by the user.
    """
    # get config of run name
    config_path = os.path.join("run_logs", run_name, "settings", "settings.yaml")
    config = utils.load_yaml_config(config_path) # config file of the run
    utils.get_device(config)

    # set logging level and format
    utils.set_logger_settings(config=config)

    # load the model
    model = load_pretrained_model(config, run_name)
    model.eval()

    # get tokenizers
    en_tokenizer, de_tokenizer = get_tokenizers()

    print("Please provide an english sentence:")
    en_sentence = input("English sentence: ")

    # Translate the sentence
    de_sentence = greedy_translate(
        en_sentence=en_sentence, 
        model=model, 
        en_tokenizer=en_tokenizer, 
        de_tokenizer=de_tokenizer, 
        config=config
        )

    sep = '-'*100
    out_str = f"\n{sep}\n\nEnglish sentence: {en_sentence}\n\nGerman sentence: {de_sentence}\n\n{sep}\n"
    print(out_str)