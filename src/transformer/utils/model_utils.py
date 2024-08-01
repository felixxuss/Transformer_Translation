from src.transformer.model import Transformer
import os
import torch
import logging

logger = logging.getLogger(__name__)

def load_new_model(config):
    # Create the model
    model = Transformer(
        vocab_size=config["VOCAB_SIZE"],
        d_model=config["D_MODEL"],
        n_heads=config["N_HEADS"],
        num_encoder_layers=config["N_ENCODER_LAYERS"],
        num_decoder_layers=config["N_DECODER_LAYERS"],
        dim_feedforward=config["DIM_FEEDFORWARD"],
        dropout=config["DROPOUT"],
        max_len=config["MAX_LEN"],
        weight_decay=config["WEIGHT_DECAY"]
    ).to(config["DEVICE"])
    logger.info(f"Loaded model with {model.parameter_count():,} parameters")

    return model

def load_pretrained_model(config, run_name):
    model = load_new_model(config)
    model = load_model_parameters(model, run_name=run_name, device=config["DEVICE"])
    return model

def load_model_parameters(model, run_name=None, device="cpu"):
    model_folder_path = os.path.join("run_logs", run_name, "models")
    model_name = os.listdir(model_folder_path)[0]
    model_path = os.path.join(model_folder_path, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model