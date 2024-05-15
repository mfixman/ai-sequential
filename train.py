import torch
import logging

from utils import load_config
from logger import Logger
from Trainer import Trainer

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description = 'Summarise this')

	parser.add_argument('--model-name', type = str, help = 'Name of the model file.')
	parser.add_argument('--model-version', type = str, help = 'Version (which is a string)', dest = 'version')
	parser.add_argument('--encoder-embedding-dim', type = int, help = 'Embedding dimensions of the encoder.')
	parser.add_argument('--decoder-embedding-dim', type = int, help = 'Embedding dimensions of the decoder.')
	parser.add_argument('--hidden-dim', type = int, help = 'Hidden dimensions of the layers of the model.')
	parser.add_argument('--num-layers', type = int, help = 'Amount of layers of the model.')
	parser.add_argument('--dropout', type = float, help = 'Dropout.')

	args = vars(parser.parse_args())
	args = {k: v for k, v in args.items() if v is not None}

	return args

def main():
	config = load_config()
	config['model_params'] |= parse_args()

	logging.basicConfig(
		level = logging.INFO,
		format = '[%(asctime)s] %(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)

	logging.info(f"Using model {config['model_params']['model_name']}")

	data_setting = config['data_settings']
	model_setting = config['model_params']
	train_setting = config['train']

	if train_setting['log']:
		wandb_logger = Logger(
f"{model_setting['model_name']}_v{model_setting['version']}_lr={train_setting['learning_rate']}_L1",
			project='NewSum',
			config = config,
		)
		logger = wandb_logger.get_logger()
	else:
		logger = None

	torch.manual_seed(42)
	trainer = Trainer(data_setting, model_setting, train_setting, logger)
	trainer.train()

if __name__ == '__main__':
	main()
