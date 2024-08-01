from torch.nn.functional import softmax
import torch
import os
import logging

logger = logging.getLogger(__name__)

def get_single_translation(data_loader, model, eval_type="train", epoch=-1, config=None):
    # take one random sentence outof the batch and translate it
    de_batch, de_mask, en_batch, en_mask = next(iter(data_loader))
    idx = torch.randint(0, config["BATCH_SIZE"], (1,)).item()
    en_sentence = en_batch[idx].tolist()
    de_sentence = de_batch[idx].tolist()
    de_mask = de_mask[idx].tolist()
    en_mask = en_mask[idx].tolist()

    # get tokenizer and get secial token ids
    en_tokenizer, de_tokenizer = data_loader.dataset.en_tokenizer, data_loader.dataset.de_tokenizer

    # translate the sentence
    de_sentence = greedy_translate(en_sentence, model, en_tokenizer, de_tokenizer, config)

    # remove PAD, BOS and EOS tokens
    en_sentence = [token for token in en_sentence if token not in en_tokenizer.special_tokens.values()]    

    # decode the english sentence
    en_sentence = en_tokenizer.tokenizer.decode(en_sentence)

    # show translation from english to german
    sep = '-'*40
    out_str = f"\n{sep}  {eval_type} - Epoch {epoch}  {sep}\nEnglish sentence: {en_sentence}\nGerman sentence: {de_sentence}\n{sep}  {eval_type} - Epoch {epoch}  {sep}\n"
    # print(out_str)

    # save translation to a text file
    translation_file = os.path.join(config["TRANSLATION_DIR"], "translations.txt")
    with open(translation_file, "a", encoding="utf-8") as f:
        f.write(out_str)

def greedy_translate(en_sentence, 
                     model, 
                     en_tokenizer, 
                     de_tokenizer, 
                     config):
    if isinstance(en_sentence, str):
        en_sentence = en_tokenizer.encode(en_sentence)

    translation = autoregressive_translation(
        model=model,
        en_sentence=en_sentence,
        bos_token=en_tokenizer.tokenizer.bos_token_id,
        eos_token=en_tokenizer.tokenizer.eos_token_id,
        pad_token=en_tokenizer.tokenizer.pad_token_id,
        config=config
    )

    # remove special tokens
    translation = translation.squeeze(0).tolist()
    translation = [token for token in translation if token not in en_tokenizer.special_tokens.values()] or [en_tokenizer.special_tokens["PAD"]]
    
    # decode the tokens
    de_sentence = de_tokenizer.decode(translation)

    return de_sentence

def autoregressive_translation(model,
                               en_sentence,
                               bos_token,
                               eos_token,
                               pad_token,
                               config):
    en_sentence = [token for token in en_sentence if token not in [bos_token, eos_token, pad_token]]
    en_tokens = [bos_token, *en_sentence, eos_token]
    en_tokens = torch.tensor(en_tokens).unsqueeze(0).to(config["DEVICE"])
    en_mask = torch.ones_like(en_tokens).to(config["DEVICE"])

    en_encoded = model.encode(en_tokens, en_mask, config)

    assert en_encoded.shape[0] == 1, "Batch size must be 1"
    
    # feed autoregressivly through the decoder and head
    translation = torch.tensor([bos_token]).unsqueeze(0).to(config["DEVICE"])
    for idx in range(config["MAX_LEN"]-1):
        translation_mask = torch.ones_like(translation).to(config["DEVICE"])
        de_encoded = model.decode(translation, translation_mask, en_encoded, en_mask, config)
        de_encoded = model.head(de_encoded)
        de_encoded = softmax(de_encoded, dim=-1)

        de_token = torch.argmax(de_encoded, dim=-1)[0,-1].item()

        if de_token == eos_token:
            logger.debug(f"Found EOS token at index {idx}")
            break
        
        de_token = torch.tensor([de_token]).unsqueeze(0).to(config["DEVICE"])
        translation = torch.cat((translation, de_token), dim=-1)
    else:
        logger.debug("Did not find EOS token")
    
    return translation