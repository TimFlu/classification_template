import logging
from comet_ml import start


def setup_comet_logger(experiment_name):
    experiment = start(
    api_key="5OzmIvJNsXYfBCTb5CTYF8Bqy",
    project_name="perormance-metric-estim",
    workspace="timflu",
    )
    experiment.set_name(experiment_name)
    return experiment

def comet_log_metrics(comet_logger, metrics, step, cfg):
    if cfg.comet_logger.initialize:
        comet_logger.log_metrics(metrics, step=step)
    else:
        return