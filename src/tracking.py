from ctypes import Union

import os
import shutil
import omegaconf
import wandb
import datetime as dt
from lightning.pytorch.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from termcolor import colored

TERM_COLOR = "cyan"


def init_logger(cfg):
    date_time = dt.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    model_prefix = f"COA_{cfg.train.coatt_loss}_SOC_{cfg.train.social_loss}_coatt_hm_{cfg.model.coatt_hm_type}"
    if cfg.model.coatt_hm_type == 'coef_iter':
        model_prefix += f"_{cfg.model.coatt_coef_iter_cnt}"
    exp_id = f'{date_time}_{model_prefix}'

    # set the exp_id in cfg
    cfg.experiment.exp_id = exp_id

    # experiment_name = f'{cfg.experiment.name}-{cfg.experiment.dataset}'
    if cfg.wandb.log:
        id = wandb.util.generate_id()  # type: ignore
        logger = WandbLogger(
            project=cfg.project.name,
            entity="chihiro-nakatani",
            group=cfg.experiment.group,
            log_model=False,
            id=id,
            # name=cfg.experiment.name,
            name=cfg.experiment.exp_id,
            save_dir="./",
            allow_val_change=True,
        )

        cfg_dict = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        if rank_zero_only.rank == 0:
            logger.experiment.config.update(cfg_dict)
    else:
        logger = False

    return logger, cfg


def save_code_snapshot(code_folder, output_folder="."):
    print(colored("Saving a snapshot of the source code folder ...", TERM_COLOR), end=" ")
    output_basename = os.path.join(output_folder, "src")
    shutil.make_archive(output_basename, "zip", code_folder)
    print(colored("Done.", TERM_COLOR))
    
