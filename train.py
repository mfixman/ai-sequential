import torch
import logging
import wandb

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

	parser.add_argument('--epochs', type = int, help = 'Amount of epochs to train from.')
	parser.add_argument('--batch-size', type = int, help = 'Batch size.')
	parser.add_argument('--learning-rate', type = float, help = 'Alpha of the model.')
	parser.add_argument('--load-checkpoint', type = bool, default = False, help = 'Start loading a checkpoint.')
	parser.add_argument('--log', type = bool, default = True, help = 'Log some data in Hyperion,')
	parser.add_argument('--max-samples', type = int, help = 'Training samples to take data from.')
	parser.add_argument('--varkappa', type = float, help = 'Weight of Cosine Similarity Loss against Categorical Cross Entropy Loss.')

	parser.add_argument('--train-validation-set', action = 'store_true', help = 'Use the validation set as training set. This provides incorrect results, but speeds up training.')

	parser.add_argument('--device', choices = ['cuda', 'cpu'], help = 'Device to use')

	args = vars(parser.parse_args())
	args = {k: v for k, v in args.items() if v is not None}

	return args

def main(changes = {}):
	config = load_config()
	args = parse_args()
	config['model_params'] |= args | changes
	config['train'] |= args | changes

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

	sweep_params = ['learning_rate', 'varkappa']
	change = False
	for sp in sweep_params:
		param = 'sweep_' + sp
		if param in wandb.config:
			train_setting[sp] = wandb.config[param]
			model_setting[sp] = wandb.config[param]
			change = True
	if change:
		logging.info(f'New settings for param sweep: {train_setting} {model_setting}')

	torch.manual_seed(42)
	trainer = Trainer(data_setting, model_setting, train_setting, logger, device = args.get('device', None))
	trainer.train()

if __name__ == '__main__':
	main()
