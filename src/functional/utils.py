import yaml
import logging
import os
import time
import shutil
import torch 

def load_yaml_config(settings_path):
    with open(settings_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_logger_settings(config):
    logging.basicConfig(level=config["log_level"], format=config["log_format"], datefmt="%H:%M:%S")
    # set global logging level
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if "modelling" in name]
    for logger in loggers:
        logger.setLevel(config["log_level"])

def create_file_structure(config):
    # set device
    get_device(config)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config['TRAIN_RUN_NAME']}_{timestamp}"

    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    logs_dir = os.path.join(dir_path, "run_logs")

    # create folder if not exists
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # create run directory
    run_directory = os.path.join(logs_dir, run_name)
    os.makedirs(run_directory)

    # create plot directory
    plot_directory = os.path.join(run_directory, "plots")
    config["PLOT_DIR"] = plot_directory
    os.makedirs(plot_directory)

    # create model directory
    model_directory = os.path.join(run_directory, "models")
    config["MODEL_DIR"] = model_directory
    os.makedirs(model_directory)

    # crate translation directory
    translation_directory = os.path.join(run_directory, "translations")
    config["TRANSLATION_DIR"] = translation_directory
    os.makedirs(translation_directory)
    # create empty translation text file
    translation_file = os.path.join(translation_directory, "translations.txt")
    with open(translation_file, "w") as f:
        f.write("")
        
    # create settings directory
    settings_directory = os.path.join(run_directory, "settings")
    os.makedirs(settings_directory)

    # copy settings file to settings directory
    settings_file = os.path.join(dir_path, "settings.yaml")
    shutil.copy(settings_file, settings_directory)

def get_device(config):
    # save torch device to config file
    config["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

