import argparse
import src.functional.utils as utils

def main(args):
    # get config file
    config = utils.load_yaml_config("settings.yaml")

    # set logging level and format
    utils.set_logger_settings(config)

    if args.train:
        from src.functional.trainer import train
        utils.create_file_structure(config)
        train(config)

    elif args.sample:
        from src.functional.sampler import sample
        sample(config["SAMPLE_RUN_NAME"])

    elif args.bleu:
        from src.functional.bleu_score import eval_model
        eval_model(config["SAMPLE_RUN_NAME"])

    else:
        raise ValueError("Set either --train, --sample or --bleu.")
    

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='If set, a new model will be trained')
    parser.add_argument('--sample', action='store_true', help='If set, the user can translate a single sentence.')
    parser.add_argument('--bleu', action='store_true', help='If set, the sample model will be evaluated')
    args = parser.parse_args()

    # Call main function with mode argument
    main(args)