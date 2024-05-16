import torch
import logging

from utils import load_config
from logger import Logger
from Trainer import Trainer

import wandb
import argparse

import train

def betterMain():
	train.main(dict(
		model_name = 'transformer',
		version = '2',
		epochs = 5,
		max_samples = 10,
		train_validation_set = True,
	))

def main():
	sweep_configuration = dict(
		name = 'fun_sweep',
		method = 'grid',
		parameters = {
			'sweep_learning_rate': {'values': [0.01, 0.001, 0.0001]},
			'sweep_varkappa': {'values': [0, 0.2, 0.5]},
			# 'sweep_dropout': [0, 0.1, 0.25],
		}
	)
	sweep = wandb.sweep(sweep = sweep_configuration, entity = 'aabati', project = 'NewSum')
	wandb.agent(sweep, function = betterMain, count = 50)

if __name__ == '__main__':
	main()
