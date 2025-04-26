import os
import torch
from core.logger.logger import setup_logger
from core.logger.tensorboard_logger import TensorboardLogger
from core.utils.distributed_utils import init_distributed_mode
from core.utils.args import Configuration
from core.manager.trainer import Trainer
from core.utils.util import setup_seed


def main(config):
    if config['manual_seed'] is not None:
        setup_seed(config['manual_seed'], config['seed_deterministic'])
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    config['image_save_dir'] = f"./log/{config['model_name']}/{config['stage']}/image"
    os.makedirs(config['image_save_dir'], exist_ok=True)
    config['distributed'] = True if torch.cuda.device_count() > 1 else False
    if config['distributed']:
        init_distributed_mode(config)
        rank = config['rank']
        if rank == 0:
            logger = setup_logger(os.path.join(config['exp'], config['save_path'], 'training.log'))
            logger.info(config)
            logger.info("Start Tensorboard with tensorboard --logdir=log, view at http://localhost:6006/")
            tb_writer = TensorboardLogger(log_path=f"log/{config['model_name']}/{config['stage']}")
            trainer = Trainer(config=config, logger=logger, writer=tb_writer, local_rank=rank)
        else:
            trainer = Trainer(config=config, local_rank=rank)
    else:
        logger = setup_logger(os.path.join(config['exp'], config['save_path'], 'training.log'))
        logger.info(config)
        logger.info("Start Tensorboard with tensorboard --logdir=log, view at http://localhost:6006/")
        tb_writer = TensorboardLogger(log_path=f"log/{config['model_name']}/{config['stage']}")
        trainer = Trainer(config=config, logger=logger, writer=tb_writer, local_rank=0)
    trainer.clip_training()


if __name__ == '__main__':
    config = Configuration()
    config.get_parser()
    main(config)
