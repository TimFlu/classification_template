from utils.training import train_model
from utils.log import setup_comet_logger
import torch
import logging
logger = logging.getLogger(__name__)
import hydra
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# TODO: Config name
@hydra.main(config_path="config", config_name="config_template")
def main(cfg):
    # Setup comet logger
    if cfg.comet_logger.initialize:
        logger.info("Setting up comet logger")
        config_name = HydraConfig.get().job.config_name
        comet_logger = setup_comet_logger(experiment_name=config_name)
        comet_logger.log_parameters(OmegaConf.to_container(cfg))
    else:
        comet_logger = None
    initial_dir = hydra.utils.get_original_cwd()
    logger.info("Initial dir: {}".format(initial_dir))
    logger.info("Current dir: {}".format(os.getcwd()))
    logger.info('Current config: ')
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Train the model
    train_model(device, comet_logger, cfg)

if __name__ == "__main__":
    main()