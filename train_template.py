from utils.training import train_model
from utils.log import setup_comet_logger
import torch
import logging
logger = logging.getLogger(__name__)
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# TODO: Config name
@hydra.main(config_path="config", config_name="config_template")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))

    # Setup comet logger
    if cfg.comet_logger.initialize:
        logger.info("Setting up comet logger")
        script_name = HydraConfig.get().job.name
        comet_logger = setup_comet_logger(experiment_name=script_name)
        comet_logger.log_parameters(OmegaConf.to_container(cfg))
    else:
        comet_logger = None
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Train the model
    train_model(device, comet_logger, cfg)

if __name__ == "__main__":
    main()