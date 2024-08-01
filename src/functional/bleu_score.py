import os
from tqdm.auto import tqdm
import evaluate
import logging

# modules
import src.functional.utils as utils
from src.transformer.utils.model_utils import load_pretrained_model
from src.transformer.utils.tokenizer import get_tokenizers
from src.functional.data_helper.data_utils import load_translation_data
from src.functional.sampler_helper.sample_utils import greedy_translate

logger = logging.getLogger(__name__)

def eval_model(run_name=None, mode="test"):
    """Method to calculate the BLEU score for a pretrained model.
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

    bleu_scores = calc_bleu(model, config, mode=mode)
    
    print(f"\nBLEU scores for {mode} data:")
    for i, score in enumerate(bleu_scores):
        print(f"BLEU-{i+1}: {score['bleu']:.4f}")


def calc_bleu(model, config, mode="test"):
    # get the data
    data_pairs = load_translation_data(mode)

    # print number of samples
    logger.info(f"{len(data_pairs)} samples from {mode} data used for BLEU calculation.")

    # Get the tokenizer
    en_tokenizer, de_tokenizer = get_tokenizers()

    predictions = []
    references = []
    it = tqdm(data_pairs, desc="Calculating BLEU")
    for pair in it:
        de_sentence, en_sentence = pair["de"], pair["en"]
        # Translate the sentence
        de_translation = greedy_translate(
            en_sentence=en_sentence, 
            model=model, 
            en_tokenizer=en_tokenizer, 
            de_tokenizer=de_tokenizer, 
            config=config
            )
        predictions.append(de_translation)
        references.append([de_sentence])
        
    bleu = evaluate.load('bleu')
    bleu_scores = []
    for order in range(1,5):
        score = bleu.compute(predictions=predictions, references=references, 
            max_order = order)
        bleu_scores.append(score)

    return bleu_scores