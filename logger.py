import wandb

class Logger:
    def __init__(self, experiment_name, logger_name='logger', project='inm70x', config = None):
        logger_name = f'{logger_name}-{experiment_name}'
        logger = wandb.init(project=project, name=logger_name, config = config)
        self.logger = logger
        return

    def get_logger(self):
        return self.logger
