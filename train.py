import torch
import logging

from utils import load_config
from logger import Logger
from Trainer import Trainer

def main():
	config = load_config()

	logging.basicConfig(
		level = logging.INFO,
		format = '[%(asctime)s] %(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)

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
