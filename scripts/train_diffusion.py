"""Train diffusion-based Neural A*.

The DDPM learns a bright optimal-path heatmap from the same mazes dataset used
by Neural A*. During validation, that heatmap is flipped into a cost map and
sent through the original differentiable A* module.
"""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import DiffusionAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import DiffusionPlannerModule, set_global_seeds
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


@hydra.main(version_base=None, config_path="config", config_name="train_diffusion")
def main(config):
    set_global_seeds(config.seed)
    train_loader = create_dataloader(
        config.dataset + ".npz",
        "train",
        config.params.batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        config.dataset + ".npz",
        "valid",
        config.params.batch_size,
        shuffle=False,
    )

    planner = DiffusionAstar(
        unet_config=OmegaConf.to_container(config.unet, resolve=True),
        diffusion_config={
            k: v
            for k, v in OmegaConf.to_container(config.diffusion, resolve=True).items()
            if k not in {"trajectory_sigma", "start_goal_sigma"}
        },
        Tmax=config.Tmax,
        start_goal_sigma=config.diffusion.start_goal_sigma,
    )
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor=config.checkpoint.monitor,
        save_weights_only=True,
        mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k,
        save_last=config.checkpoint.save_last,
    )
    callbacks.append(checkpoint_callback)
    if config.early_stop.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=config.early_stop.monitor,
                mode=config.early_stop.mode,
                patience=config.early_stop.patience,
                min_delta=config.early_stop.min_delta,
                verbose=True,
            )
        )

    module = DiffusionPlannerModule(planner, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}_diffusion"
    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=callbacks,
        limit_train_batches=config.params.limit_train_batches,
        limit_val_batches=config.params.limit_val_batches,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    )
    if config.trainer.devices is not None:
        trainer_kwargs["devices"] = config.trainer.devices
    if config.trainer.strategy is not None:
        trainer_kwargs["strategy"] = config.trainer.strategy
    if config.trainer.precision is not None:
        trainer_kwargs["precision"] = config.trainer.precision
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_loader, val_loader)
    print(f"best_model_path={checkpoint_callback.best_model_path}")
    print(f"best_model_score={checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    main()
