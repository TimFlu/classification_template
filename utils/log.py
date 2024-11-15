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

def comet_log_figure(comet_logger, figure, name, step, cfg):
    if cfg.comet_logger.initialize:
        comet_logger.log_figure(f'{name}_epoch_{step}', figure, step=step)
    else:
        return