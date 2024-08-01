# torch imports
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# other imports
import os
import time
import logging
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

# modules
from src.transformer.utils.model_utils import load_new_model
from src.transformer.utils.lr_scheduler import LRScheduler
from src.functional.data_helper.data_utils import get_dataloader
from src.functional.sampler_helper.sample_utils import get_single_translation

# get logger
logger = logging.getLogger(__name__)

def train(config):
    # load the model
    model = load_new_model(config)

    # get data loaders
    train_data_loader = get_dataloader("train", subset_n_batches=config["N_TRAINING_BATCHES"], config=config)
    val_data_loader = get_dataloader("val", subset_n_batches=config["N_TRAINING_BATCHES"], config=config)
    
    # train the model
    train_model(model, train_data_loader, val_data_loader, config)


def train_model(model, train_data_loader, val_data_loader, config):
    """Trains the model

    Args:
        model (Transformer): The model to train
        train_data_loader (Dataloader): The training data loader
        val_data_loader (Dataloader): The validation data loader
        save_loss_plot (bool, optional): Whether the loss plots should be saved. Defaults to True.
        save_model (bool, optional): Whether the model should be saved. Defaults to True.
    """
    logger.info("Start training")
    start_time = time.perf_counter()
    optimizer = AdamW(model.param_groups)
    scheduler = LRScheduler(
        optimizer, 
        d_model=config["D_MODEL"], 
        warmup_steps=config["WARMUP_STEPS"],
        step_size=config["STEP_SIZE"]
        )
    
    loss_fn = CrossEntropyLoss(ignore_index=0, label_smoothing=config["LABEL_SMOOTHING"]).to(config["DEVICE"])

    train_losses = []
    val_losses = []

    logger.debug(f"Training on device: {config['DEVICE']}")
    for epoch in range(1,config["EPOCHS"]+1):
        logger.info(f"Epoch {epoch}/{config['EPOCHS']}")

        # train model
        mean_loss = train_epoch(model, optimizer, loss_fn, train_data_loader, config)
        train_losses.append(mean_loss)

        # validate model
        val_loss = validate_model(model, val_data_loader, loss_fn=loss_fn, config=config)
        val_losses.append(val_loss)
        
        # take a step after every epoch
        scheduler.step()

        # sample a random translation from the training and validation set
        with torch.no_grad():
            # get one translation from the training set
            get_single_translation(train_data_loader, model, eval_type="train", epoch=epoch, config=config)

            # get single translation from the validation set
            get_single_translation(val_data_loader, model, eval_type="val", epoch=epoch, config=config)
        

        logger.info(f"Epoch {epoch}/{config['EPOCHS']} - train loss: {mean_loss:.4f} - val loss: {val_loss:.4f}")

        # plot and save to png with timestamp
        save_loss(train_losses, val_losses, epoch, config=config)

    end_time = time.perf_counter()
    logger.info(f"Training took {(end_time-start_time)/60:.2f} minutes")

    # save model
    save_model(model, config)

    logger.info("End training")

def train_epoch(model, optimizer, loss_fn, data_loader, config):
    # bring model to train mode
    model.train()
    mean_loss = 0
    batch_iterator = tqdm(data_loader)
    batch_iterator.set_description(f"Training")
    for de_batch, de_mask, en_batch, en_mask in batch_iterator:
        # input slicing
        de_batch_input, de_mask_input = de_batch[:, :-1], de_mask[:, :-1] # cut of <EOS> token -> use for forward pass
        de_batch_target = de_batch[:, 1:] # cut of <BOS> token -> use for loss

        # shift tensors to right device
        en_batch, en_mask = en_batch.to(config["DEVICE"]), en_mask.to(config["DEVICE"])
        de_batch_input, de_mask_input = de_batch_input.to(config["DEVICE"]), de_mask_input.to(config["DEVICE"])
        de_batch_target = de_batch_target.to(config["DEVICE"])

        # translation from english to german
        out = model(
            en_batch=en_batch, 
            de_batch=de_batch_input, 
            en_mask=en_mask, 
            de_mask=de_mask_input, 
            config=config) # (batch_size, context_length, vocab_size)
    
        # calculate loss
        out, de_batch_target = out.view(-1, out.shape[-1]), de_batch_target.reshape(-1)
        loss = loss_fn(out, de_batch_target)
        mean_loss += loss.item()

        # zero gradients
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # update model parameters
        optimizer.step()

    # calculate mean loss
    mean_loss /= len(data_loader)

    return mean_loss

def validate_model(model, data_loader, loss_fn, config):
    model.eval()
    mean_loss = 0
    with torch.no_grad():
        batch_iterator = tqdm(data_loader)
        batch_iterator.set_description(f"Validation")
        for de_batch, de_mask, en_batch, en_mask in batch_iterator:
            # input slicing
            de_batch_input, de_mask_input = de_batch[:, :-1], de_mask[:, :-1] # cut of <EOS> token -> use for forward pass
            de_batch_target = de_batch[:, 1:] # cut of <BOS> token -> use for loss

            # shift tensors to right device
            en_batch, en_mask = en_batch.to(config["DEVICE"]), en_mask.to(config["DEVICE"])
            de_batch_input, de_mask_input = de_batch_input.to(config["DEVICE"]), de_mask_input.to(config["DEVICE"])
            de_batch_target = de_batch_target.to(config["DEVICE"])

            out = model(
                en_batch=en_batch, 
                de_batch=de_batch_input, 
                en_mask=en_mask, 
                de_mask=de_mask_input, 
                config=config) # (batch_size, context_length, vocab_size)

            # calculate loss
            out, de_batch_target = out.view(-1, out.shape[-1]), de_batch_target.reshape(-1)
            loss = loss_fn(out, de_batch_target)
            mean_loss += loss.item()
            
    mean_loss /= len(data_loader)

    return mean_loss

def save_model(model, config):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"model_{timestamp}.pt"
    save_path = os.path.join(config["MODEL_DIR"], file_name)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved the model")


def save_loss(train_losses: list[float], val_losses: list[float], epoch: int, config=None):
    """Saves the loss plot

    Args:
        train_losses (list[float]): List of training losses
        val_losses (list[float]): List of validation losses
        epoch (int): The current epoch
    """
    # save 5 plots in total
    if epoch > 1:
        plot_interval = config["EPOCHS"] // 5 if config["EPOCHS"] > 5 else 1
        if not epoch % (plot_interval):
            # controll external loggers
            logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
            logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
            logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)

            plt.figure(figsize=(15,5))
            plt.plot(train_losses, label="train loss", color="black")
            plt.plot(val_losses, label="val loss", color="red")
            plt.yticks(np.arange(0, max(train_losses+val_losses)+2))
            plt.xticks(np.arange(0, epoch+1))
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            file_name = f"loss_epoch{epoch}_{time.strftime('%H-%M-%S')}.png"
            save_path = os.path.join(config["PLOT_DIR"], file_name)
            plt.savefig(save_path)
            plt.close()