import datetime as dt
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd
import numpy as np
import lightning.pytorch as pl
import torch

# from pytorch_lightning.profiler import AdvancedProfiler
from lightning.pytorch.callbacks import BaseFinetuning, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from termcolor import colored

from src.datasets.gazefollow import GazeFollowDataModule
from src.datasets.combined_social import CombinedSocialDataModule
from src.datasets.videoattentiontarget_temporal import VideoAttentionTargetDataModule
from src.datasets.videocoatt_temporal import VideoCoAttDataModule
from src.datasets.uco_laeo_temporal import VideoLAEODataModule
from src.datasets.childplay_temporal import ChildPlayDataModule
from src.datasets.vacation import VacationDataModule
from src.models import BaselineModel, ChongModel, NoraModel, SharinganModel, InteractModel
from src.models_custom import GazeLLE
from src.tracking import init_logger, save_code_snapshot
from src.utils import Stage
import sys
import os
import yaml
import omegaconf

TERM_COLOR = "cyan"


# ============================================================================================================ #
#                                             BASE EXPERIMENT CLASS                                            #
# ============================================================================================================ #
class BaseExperiment(ABC):
    """Base class for experiments."""

    @abstractmethod
    def parse_experiment(self, experiment):
        """Parse experiment string."""
        pass

    @abstractmethod
    def setup(self):
        """Setup experiment (e.g. create callbacks, init losses and metrics etc.)."""
        pass

    @abstractmethod
    def run(self):
        """Run experiment."""
        pass


# ============================================================================================================ #
#                                                 EXPERIMENT CLASS                                             #
# ============================================================================================================ #
class Experiment(BaseExperiment):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tasks = self.parse_experiment(self.cfg.experiment.task)

    def parse_experiment(self, experiment):
        return experiment.split("+")

    def set_seed(self):
        if self.cfg.train.seed is not None:
            pl.seed_everything(self.cfg.train.seed)

    def init_model(self):
        # model = RinneganModel(cfg=self.cfg)
        # model = MultiViTRinneganModel(cfg=self.cfg)
        if self.cfg.model.model_name == 'gaze_lle':
            model = GazeLLE(cfg=self.cfg)
        else:
            model = InteractModel(cfg=self.cfg)
        # model = SharinganModel(cfg=self.cfg)
        # model = ChongModel(cfg=self.cfg)
        # model = BaselineModel(cfg=self.cfg)
        # model = NoraModel(cfg=self.cfg)
        # model = torch.compile(model) 
        return model

    def init_data(self):
        if self.cfg.experiment.dataset == "gazefollow":
            data = GazeFollowDataModule(
                root=self.cfg.data.gf.root,
                root_depth=self.cfg.data.gf.root_depth,
                root_focal=self.cfg.data.gf.root_focal,
                batch_size={
                    "train": self.cfg.train.batch_size,
                    "val": self.cfg.val.batch_size,
                    "test": self.cfg.test.batch_size,
                    "predict": self.cfg.predict.batch_size,
                },
                image_size=self.cfg.data.image_size,
                heatmap_size=self.cfg.data.heatmap_size,
                num_people = {
                    "train": self.cfg.data.num_people, 
                    "val": self.cfg.data.num_people, 
                    "test": 'all'
                },
                return_depth=self.cfg.data.return_depth,
                return_head_mask=self.cfg.data.return_head_mask,
            )
        elif self.cfg.experiment.dataset == "childplay":
            data = ChildPlayDataModule(
                cfg = self.cfg,
                root = self.cfg.data.childplay.root,
                root_depth=self.cfg.data.childplay.root_depth,
                root_focal=self.cfg.data.childplay.root_focal,
                batch_size = {
                    Stage.TRAIN: self.cfg.train.batch_size,
                    Stage.VAL: self.cfg.val.batch_size,
                    Stage.TEST: self.cfg.test.batch_size,
                    Stage.PREDICT: self.cfg.predict.batch_size,
                },
                image_size=self.cfg.data.image_size,
                heatmap_size=self.cfg.data.heatmap_size,
                num_people = {
                    "train": self.cfg.data.num_people, 
                    "val": self.cfg.data.num_people, 
                    "test": 'all'
                },
                temporal_context = self.cfg.data.temporal_context,
                temporal_stride = self.cfg.data.temporal_stride,
                return_depth=self.cfg.data.return_depth,
                dim_vlm = self.cfg.data.vlm_dim
            )
        elif self.cfg.experiment.dataset == "videoattentiontarget":
            data = VideoAttentionTargetDataModule(
                cfg = self.cfg,
                root = self.cfg.data.vat.root,
                batch_size = {
                    Stage.TRAIN: self.cfg.train.batch_size,
                    Stage.VAL: self.cfg.val.batch_size,
                    Stage.TEST: self.cfg.test.batch_size,
                    Stage.PREDICT: self.cfg.predict.batch_size,
                },
                num_people = {
                    "train": self.cfg.data.num_people, 
                    "val": self.cfg.data.num_people, 
                    "test": 'all'
                },
                temporal_context = self.cfg.data.temporal_context,
                temporal_stride = self.cfg.data.temporal_stride,
                image_size=self.cfg.data.image_size,
            )
        elif self.cfg.experiment.dataset == "videocoatt":
            data = VideoCoAttDataModule(
                root = self.cfg.data.coatt.root,
                cfg = self.cfg,
                batch_size = {
                    Stage.TRAIN: self.cfg.train.batch_size,
                    Stage.VAL: self.cfg.val.batch_size,
                    Stage.TEST: self.cfg.test.batch_size,
                    Stage.PREDICT: self.cfg.predict.batch_size,
                },
                num_people = {
                    "train": self.cfg.data.num_people, 
                    "val": self.cfg.data.num_people, 
                    # "test": 'all'
                    'test': self.cfg.data.num_people
                },
                temporal_context = self.cfg.data.temporal_context,
                temporal_stride = self.cfg.data.temporal_stride,
            )
        elif self.cfg.experiment.dataset == "uco_laeo":
            data = VideoLAEODataModule(
                cfg = self.cfg,
                root = self.cfg.data.laeo.root,
                batch_size = {
                    Stage.TRAIN: self.cfg.train.batch_size,
                    Stage.VAL: self.cfg.val.batch_size,
                    Stage.TEST: self.cfg.test.batch_size,
                    Stage.PREDICT: self.cfg.predict.batch_size,
                },
                num_people = {
                    "train": self.cfg.data.num_people, 
                    "val": self.cfg.data.num_people, 
                    "test": 'all'
                },
                temporal_context = self.cfg.data.temporal_context,
                temporal_stride = self.cfg.data.temporal_stride
            )
        elif self.cfg.experiment.dataset == "combined_social":
            data = CombinedSocialDataModule(
                cfg = self.cfg,
                root_gf = self.cfg.data.gf.root,
                root_coatt = self.cfg.data.coatt.root,
                root_laeo = self.cfg.data.laeo.root,
                root_vat = self.cfg.data.vat.root,
                root_childplay = self.cfg.data.childplay.root,
                batch_size = {
                    Stage.TRAIN: self.cfg.train.batch_size,
                    Stage.VAL: self.cfg.val.batch_size,
                    Stage.TEST: self.cfg.test.batch_size,
                    Stage.PREDICT: self.cfg.predict.batch_size,
                },
                num_people = {
                    "train": self.cfg.data.num_people, 
                    "val": self.cfg.data.num_people, 
                    "test": 'all'
                },
                temporal_context = self.cfg.data.temporal_context,
                temporal_stride = self.cfg.data.temporal_stride,
                image_size=self.cfg.data.image_size,
            )
        elif self.cfg.experiment.dataset == "vacation":
            data = VacationDataModule(
                root = self.cfg.data.vacation.root,
                batch_size = {
                    Stage.TRAIN: self.cfg.train.batch_size,
                    Stage.VAL: self.cfg.val.batch_size,
                    Stage.TEST: self.cfg.test.batch_size,
                    Stage.PREDICT: self.cfg.predict.batch_size,
                },
                num_people = {
                    "train": self.cfg.data.num_people, 
                    "val": self.cfg.data.num_people, 
                    "test": 'all'
                }
            )
        else:
            raise ValueError(
                f"Expected config.experiment.dataset to be one of [combined_social, gazefollow]. Got {self.cfg.experiment.dataset} instead."
            )
        print(colored(f"Using the {self.cfg.experiment.dataset.upper()} dataset.", TERM_COLOR))
        return data

    def init_callbacks(self):
        
        callbacks = []

        checkpoint_cb = ModelCheckpoint(
            # dirpath=f"./checkpoints/{self.cfg.experiment.dataset}/{date_time}_{model_prefix}",
            dirpath=f"./checkpoints/{self.cfg.experiment.dataset}/{self.cfg.experiment.exp_id}",
            filename="best",  # custom: "{epoch:02d}-{step:02d}-{val_acc:.3f}",
            monitor="metric/val/dist",  # "metric/val/ap", "metric/val/acc", "loss/val"
            mode="min",  # "min", "max"
            save_last=True,
            save_top_k=1,
            save_on_train_epoch_end=False,
            verbose=True,
        )
        callbacks.append(checkpoint_cb)

        # Learning Rate Monitor
        if self.cfg.wandb.log:
            lr_monitor_callback = LearningRateMonitor(logging_interval="step", log_momentum=False)
            callbacks.append(lr_monitor_callback)

        # Stochastic Weight Averaging
        if self.cfg.train.swa.use:
            swa_lrs = np.array(self.cfg.train.swa.lr)
            if len(swa_lrs.shape)>0:
                swa_lrs = swa_lrs.tolist()
            else:
                swa_lrs = swa_lrs.item()
            print(colored(f"Using Stochastic Weight Averaging (SWA).", TERM_COLOR))
            swa_callback = StochasticWeightAveraging(
                swa_lrs=swa_lrs,
                swa_epoch_start=self.cfg.train.swa.epoch_start,
                annealing_epochs=self.cfg.train.swa.annealing_epochs,
                # device=None # using gpu may overflow the gpu memory
            )
            callbacks.append(swa_callback)
            
        # Base Fine Tuning
        #bft_callback = PretrainedFreezeUnfreeze(unfreeze_at_epoch=10)
        #callbacks.append(bft_callback)

        # Save cfg as a yaml file
        save_yaml_file = os.path.join(checkpoint_cb.dirpath, "config.yaml")
        os.makedirs(checkpoint_cb.dirpath, exist_ok=True)
        with open(save_yaml_file, 'w') as f:
            yaml.dump(omegaconf.OmegaConf.to_container(self.cfg, resolve=True), f)
        print(colored(f"Saved config file to: {save_yaml_file}", TERM_COLOR))

        return callbacks

    def init_trainer(self, logger, callbacks):
        profiler = None  # AdvancedProfiler(dirpath='.', filename='advanced-profile', line_count_restriction=1.0) #None
        trainer = pl.Trainer(
            accelerator="gpu" if self.cfg.train.device == "cuda" else "auto",
            devices=1,
            precision=self.cfg.train.precision,  # 64 (double), 32 (full), 16 (16bit mixed), bf16-mixed (bfloat16 mixed). Defaults to 32.
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=False,  # uncover bugs without any lengthy training by running all the code. Doesn't generate logs or checkpoints.
            max_epochs=self.cfg.train.epochs,
            overfit_batches=0.0,  # overfit one or a few batches to find bugs. Set it to 0 to disable.
            val_check_interval=1.0,  # int for nb of batches or float in [0., 1.] for fraction of the training epoch.
            check_val_every_n_epoch=1,  # Use None to validate every n batches through `val_check_interval`. default is 1.
            num_sanity_val_steps=2,  # Sanity check runs n val batches before the training routine. Set to -1 to run all batches.
            enable_checkpointing=True,  # If True, enable checkpointing. Configures a default one if there is no ModelCheckpoint callback.
            enable_progress_bar=True,  # Whether to enable to progress bar
            enable_model_summary=True,  # Whether to enable model summarization
            accumulate_grad_batches=self.cfg.train.accumulate_grad_batches,  # accumulate gradients every k batches
            gradient_clip_val=None,  # clip gradients to this value
            gradient_clip_algorithm=None,  # "value" or "norm"
            deterministic=False,  # guarantee reproducible results by removing most of the randomness from training, but may slow it down.
            benchmark=True,  # set to True to speed up training if the input sizes for your model are fixed (e.g. during inference).
            inference_mode=False,  # Whether to use torch.inference_mode() or torch.no_grad() during evaluation (ie. validate/test/predict)
            profiler=profiler,  # None, "simple" or "advanced" to identify bottlenecks
            detect_anomaly=False,  # Enable anomaly detection for the autograd engine,
        )
        return trainer

    def setup(self):
        print(colored("Setting up the experiment ...", TERM_COLOR))
        
        # SET SEED
        self.set_seed()
        
        # SET MATMUL PRECISION
        torch.set_float32_matmul_precision(self.cfg.train.matmul_precision)

        # SAVE CODE SNAPSHOT
        #save_code_snapshot("../../../src", output_folder=".") # done in the bash script

        # INIT DATA MODULE
        self.data = self.init_data()

        # INIT LIGHTNING MODULE
        self.model = self.init_model()

        # INIT LOGGER
        self.logger, self.cfg = init_logger(cfg=self.cfg) if ("train" in self.tasks) else (None, self.cfg)

        # INIT CALLBACKS
        self.callbacks = self.init_callbacks() if ("train" in self.tasks) else None

        # INIT TRAINER
        self.trainer = self.init_trainer(self.logger, self.callbacks)

    def train(self):
        # Log model parameters and/or gradients
        if (self.cfg.wandb.log) and (self.cfg.wandb.watch is not None):
            print(colored(f"Tracking model enabled: {self.cfg.wandb.watch}.", TERM_COLOR))
            self.logger.watch(self.model, log=self.cfg.wandb.watch, log_freq=self.cfg.wandb.watch_freq, log_graph=False)

        ckpt_path = self.cfg.train.resume_from if self.cfg.train.resume else None
        print(colored(f"Resuming model training from: `{ckpt_path}`.", TERM_COLOR))
        # self.trainer.fit(self.model, self.data, ckpt_path=ckpt_path)

        # if ckpt_path:
        #     # Load the entire checkpoint file into memory
        #     checkpoint = torch.load(ckpt_path, map_location="cpu")
            
        #     # 2. Load ONLY the model weights using strict=False
        #     # This will load all matching layers and ignore the missing ones.
        #     self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        #     # if some parameteres are not found in the checkpoint, print them
        #     model_keys = set(self.model.state_dict().keys())
        #     checkpoint_keys = set(checkpoint['state_dict'].keys())
        #     missing_keys = model_keys - checkpoint_keys
        #     if len(missing_keys) > 0:
        #         print(colored(f"The following model parameters were not found in the checkpoint and were initialized randomly:", "yellow"))
        #         for key in missing_keys:
        #             print(colored(f"  - {key}", "yellow"))

        if ckpt_path:
            print(f"Loading checkpoint: {ckpt_path}")
            # Load the entire checkpoint file into memory
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_state_dict = checkpoint['state_dict']

            # Get the state dict of your current model
            model_state_dict = self.model.state_dict()
            
            # Create a new state dict for weights that are compatible
            filtered_state_dict = {}
            
            print("Filtering checkpoint for compatible layers...")
            for name, param in checkpoint_state_dict.items():
                # Check if the layer exists in the current model AND has the same shape
                if name in model_state_dict and param.shape == model_state_dict[name].shape:
                    filtered_state_dict[name] = param
                else:
                    # You can add a print statement here to see which layers were skipped
                    if name in model_state_dict:
                         print(colored(f"  - Skipping {name} due to shape mismatch.", "red"))
                    else:
                         print(colored(f"  - Skipping {name} as it's not in the current model.", "yellow"))

            # Load the filtered state dict
            # Using strict=False is still good practice here
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print(colored("Model weights loaded successfully from filtered checkpoint.", "green"))

        self.trainer.fit(self.model, self.data)

    def validate(self):
        ckpt_path = self.cfg.val.checkpoint
        print(colored(f"Validating model from: `{ckpt_path}`.", TERM_COLOR))
        self.trainer.validate(self.model, self.data, ckpt_path=ckpt_path, verbose=True)

    def test(self):
        ckpt_path = self.cfg.test.checkpoint if ("train" not in self.tasks) else "best"
        print(colored(f"Testing model from: `{ckpt_path}`.", TERM_COLOR))
        self.trainer.test(self.model, self.data, ckpt_path=ckpt_path, verbose=True)

        '''
        # Convert .pt checkpoint to .ckpt for PL
        ckpt_path = self.cfg.test.checkpoint if ("train" not in self.tasks) else "best"
        keys = self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

        checkpoint_data = {
            'epoch': 0,
            'global_step': 0,
            'pytorch-lightning_version': '1.9.0',
            'state_dict': self.model.state_dict(),
            'hyper_parameters': model.hparams,s
        }
        # output_ckpt_path = os.path.join('checkpoints', 'gazelle_converted.ckpt')
        output_ckpt_path = ckpt_path.replace('.pt', '.ckpt')
        torch.save(checkpoint_data, output_ckpt_path)

        assert False, 'stop after conversion'
        # '''

    def predict(self):
        raise NotImplementedError("Predict method is not implemented yet.")

    def run(self):
        print(colored("Starting the experiment ...", TERM_COLOR))
        start = dt.datetime.now()

        if "train" in self.tasks:
            self.train()

        if ("val" in self.tasks) and ("train" not in self.tasks):  # validation is already included in training
            self.validate()

        if "test" in self.tasks:
            self.test()

        if "predict" in self.tasks:
            self.predict()

        end = dt.datetime.now()
        print(colored(f"Finished. The experiment took {end - start}.", TERM_COLOR))
    
# ============================================================================================================ #
#                                                 FINE TUNING CLASS                                            #
# ============================================================================================================ #
class PretrainedFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    #image_tokenizer, encoder, gaze_encoder, gaze_decoder
    def freeze_before_training(self, pl_module):
        # Freezing modules
        print(colored(f"Freezing the model's image_tokenizer, ViT encoder and gaze_encoder.backbone modules.", TERM_COLOR))
        self.freeze(pl_module.model.image_tokenizer, train_bn=True)
        self.freeze(pl_module.model.encoder, train_bn=True)
        self.freeze(pl_module.model.gaze_encoder.backbone, train_bn=True)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(modules=pl_module.model.image_tokenizer, optimizer=optimizer, 
                                              initial_denom_lr=10, train_bn=True)
            self.unfreeze_and_add_param_group(modules=pl_module.model.encoder, optimizer=optimizer, initial_denom_lr=10, train_bn=True)
            self.unfreeze_and_add_param_group(modules=pl_module.model.gaze_encoder.backbone, optimizer=optimizer, 
                                              initial_denom_lr=10, train_bn=True)
            print(colored(f"Unfreezing the model's image_tokenizer, ViT encoder and gaze_encoder.backbone modules.", TERM_COLOR))
