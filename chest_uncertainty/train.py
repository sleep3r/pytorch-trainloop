import os
import git
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from getpass import getuser
from socket import gethostname
from contextlib import nullcontext
from typing import Optional, Union, Callable
from collections import defaultdict

import wandb
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
import segmentation_models_pytorch as smp
from rich.progress import TaskID

from config import load_config, DLConfig, object_from_dict
from dataset import ChestDataset, OrderedDistributedSampler
from checkpoint import load_checkpoint, save_checkpoint
from utils.clip_grad import dispatch_clip_grad
from utils.distributed import init_distributed, distribute_bn, reduce_tensor
from utils.env import collect_env
from utils.logging import get_logger, status
from utils.metrics import calc_metrics, AverageMeter
from utils.path import mkdir_or_exist
from utils.cuda import NativeScaler
from utils.progress import progress, task, refresh_task, update_task
from utils.tracking import log_artifact


def prepare_exp(cfg: DLConfig) -> (dict, logging.Logger):
    """All experiment preparation stuff."""
    # init the meta dict to record some important information
    # such as environment info and seed, which will be logged
    meta = dict()

    # create work_dir
    meta = create_workdir(cfg, meta)

    # init the logger before other steps
    logger = get_logger("train", cfg, meta)

    # log env info
    meta = env_collect(cfg, meta, logger=logger)

    # set random seeds
    meta = determine_exp(cfg, meta, logger=logger)

    # setup eval metrics tracking
    meta = setup_metrics(cfg, meta, logger=logger)

    # setup w&b tracking
    setup_tracking(cfg, meta)
    return meta, logger


def create_workdir(cfg: DLConfig, meta: dict) -> dict:
    """
    Creates working directory for artifacts storage.

    Args:
        cfg (DLConfig): config object;
        meta (dict): meta dictionary.
    Returns:
        dict: updated meta dictionary.
    """
    dirname = f"{cfg.training.exp_name}/{datetime.now().strftime('%d.%m/%H.%M.%S')}"
    meta["run_name"] = dirname
    meta["exp_dir"] = Path(cfg.training.work_dir) / dirname
    mkdir_or_exist(meta["exp_dir"])
    return meta


def set_random_seed(seed: int = 228, precision: int = 10, deterministic: bool = False) -> None:
    """
    Sets random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for CUDNN backend,
                              i.e., set `torch.backends.cudnn.deterministic` to True and
                              `torch.backends.cudnn.benchmark` to False. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def env_collect(cfg: DLConfig, meta: dict, logger: logging.Logger) -> dict:
    """Collects environment information."""
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'

    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    repo = git.Repo(search_parent_directories=True)
    meta["sha"] = repo.head.object.hexsha
    meta["host_name"] = f"{getuser()}@{gethostname()}"
    return meta


def determine_exp(cfg: DLConfig, meta: dict, logger: logging.Logger) -> dict:
    """Sets seed and experiment name."""
    if cfg.training.seed is not None:
        logger.info(f'Set random seed to {cfg.training.seed}, deterministic: {cfg.training.deterministic}\n')
        set_random_seed(
            cfg.training.seed,
            precision=cfg.training.precision,
            deterministic=cfg.training.deterministic
        )

    meta['seed'] = cfg.training.seed
    meta['exp_name'] = cfg.training.exp_name
    return meta


def setup_metrics(cfg: DLConfig, meta: dict, logger: logging.Logger) -> dict:
    meta["stages"] = defaultdict(dict)

    for stage in cfg.stages:
        meta["stages"][stage]["train_loss"] = []
        meta["stages"][stage]["val_loss"] = []

        meta["stages"][stage]["lr"] = []

        meta["stages"][stage]["best_metric"] = 0
        meta["stages"][stage]["best_epoch"] = 0
    return meta


def setup_tracking(cfg: DLConfig, meta: dict) -> None:
    """Sets up mlflow tracking."""
    if cfg.local_rank == 0:
        wandb.init(
            project="chest_uncertainty",
            name=cfg.training.exp_name,
            config=cfg.to_dict()
        )


def track_metrics(
        cfg: DLConfig,
        stage: str,
        epoch: int,
        train_metrics: dict,
        eval_metrics: dict,
        meta: dict,
        logger: logging.Logger
) -> None:
    if cfg.local_rank == 0:
        dash_line = '-' * 60 + '\n'

        meta["stages"][stage]["train_loss"].append(train_metrics["loss"])
        meta["stages"][stage]["val_loss"].append(eval_metrics["loss"])

        # loss
        loss_info = f'[Epoch]: {epoch}\nTrain loss: {train_metrics["loss"]}\nVal loss: {eval_metrics["loss"]}\n'
        wandb.log({f"{stage}_train_loss": train_metrics["loss"]}, step=epoch)
        wandb.log({f"{stage}_val_loss": eval_metrics["loss"]}, step=epoch)

        # metrics
        metrics_info = '\n'.join([f'{k}: {v}' for k, v in eval_metrics.items() if k != "loss"])
        for metric in cfg.evaluation.metrics:
            wandb.log({f"{stage}_{metric}": eval_metrics[metric]}, step=epoch)

        # lr
        wandb.log({f"{stage}_lr": train_metrics["lr"]}, step=epoch)

        logger.info(dash_line + loss_info + '\n' + metrics_info + '\n' + dash_line)


def log_artifacts(cfg: DLConfig, meta: dict) -> None:
    """Logs all artifacts."""
    if cfg.local_rank == 0:
        with status(f"[bold green]Logging artifacts"):
            # dump and log config
            cfg.dump(meta["exp_dir"] / "config.yml")
            log_artifact("config", meta["exp_dir"] / "config.yml", type="config")

            # dump and log report json with meta info
            with open(meta["exp_dir"] / "report.json", "w") as f:
                meta["exp_dir"] = str(meta["exp_dir"])  # json serialization doesn't like Path
                json.dump(meta, f, indent=4)
            log_artifact("report", f"{meta['exp_dir']}/report.json", type="report")

            # dump run logfile
            log_artifact("logfile", f"{meta['exp_dir']}/run.log", type="logfile")


def get_loader_config(
        cfg: DLConfig,
        stage: str,
        dataset: ChestDataset,
        distributed: bool,
        kind: str = "train"
) -> dict:
    """Builds loader config."""
    stage_cfg = cfg.stages[stage]
    collate_fn = object_from_dict(stage_cfg.loader.get("collate_fn", {}))

    loader_cfg = {
        **stage_cfg.loader,
        **dict(
            collate_fn=collate_fn
        ),
    }
    # TRAIN
    if kind == "train":
        if distributed:
            loader_cfg["sampler"] = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True, num_replicas=cfg.world_size, rank=cfg.local_rank  # noqa
            )

    # VAL
    elif kind == "val":
        if distributed:
            loader_cfg["sampler"] = OrderedDistributedSampler(
                dataset, num_replicas=cfg.world_size, rank=cfg.local_rank
            )
        loader_cfg["drop_last"] = False
    else:
        raise ValueError("Either val or train kind must be specified")
    return loader_cfg


def eval_batch(
        cfg: DLConfig,
        outputs: torch.Tensor,
        targets: torch.Tensor,
) -> dict:
    metrics = calc_metrics(cfg, outputs, targets)
    return metrics


def val_epoch(
        model: Union[DP, DDP],
        loss_fn: Callable,
        loader: DataLoader,
        cfg: DLConfig,
        distributed: bool,
        progress_task: Optional[TaskID],
        amp_autocast=nullcontext,
):
    metrics = {}
    metrics["loss"] = AverageMeter()
    for metric in cfg.evaluation.metrics:
        metrics[metric] = AverageMeter()

    model.eval()

    refresh_task(progress_task)
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(loader):
            data_batch["image"] = data_batch["image"].cuda()
            data_batch["mask"] = data_batch["mask"].cuda()

            if cfg.training.channels_last:
                data_batch["image"] = data_batch["image"].contiguous(memory_format=torch.channels_last)  # noqa

            with amp_autocast():
                outputs = model(data_batch["image"])
                loss = loss_fn(outputs, data_batch["mask"])
                num_samples = data_batch["image"].size(0)
                batch_metrics = {}

            for metric, value in eval_batch(cfg, outputs, data_batch["mask"]).items():
                batch_metrics[metric] = value

            if distributed:
                reduced_loss = reduce_tensor(loss.data, cfg.world_size)

                for metric, value in batch_metrics.items():
                    batch_metrics[metric] = reduce_tensor(value.data, cfg.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            metrics["loss"].update(reduced_loss.item(), num_samples)
            for metric in cfg.evaluation.metrics:
                metrics[metric].update(batch_metrics[metric].item(), num_samples)

            update_task(progress_task, advance=cfg.world_size)
            # end for
        # end with

    for metric in metrics:
        metrics[metric] = metrics[metric].avg
    return metrics


def train_one_epoch(
        model: Union[DP, DDP],
        loss_fn: Callable,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        cfg: DLConfig,
        distributed: bool,
        progress_task: Optional[TaskID],
        amp_autocast=nullcontext,
        gradient_scaler: NativeScaler = None,
        logger: logging.Logger = None
):
    metrics = {}
    losses_m = AverageMeter()

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

    model.train()

    last_idx = len(loader) - 1
    refresh_task(progress_task)
    for batch_idx, data_batch in enumerate(loader):
        last_batch = batch_idx == last_idx

        data_batch["image"] = data_batch["image"].cuda()
        data_batch["mask"] = data_batch["mask"].cuda()

        if cfg.training.channels_last:
            data_batch["image"] = data_batch["image"].contiguous(memory_format=torch.channels_last)  # noqa

        with amp_autocast():
            outputs = model(data_batch["image"])
            loss = loss_fn(outputs, data_batch["mask"])
            num_samples = data_batch["image"].size(0)

        if not distributed:
            losses_m.update(loss.item(), num_samples)

        optimizer.zero_grad()
        if gradient_scaler is not None:
            gradient_scaler(
                loss, optimizer,
                clip_grad=cfg.training.clip_grad,
                clip_mode=cfg.training.clip_mode,
                parameters=model.parameters(),
                create_graph=second_order
            )
        else:
            loss.backward(create_graph=second_order)

            if cfg.training.clip_grad is not None:
                dispatch_clip_grad(
                    model.parameters(),
                    value=cfg.training.clip_grad,
                    mode=cfg.training.clip_mode
                )

            optimizer.step()

        torch.cuda.synchronize()

        if last_batch and distributed:
            reduced_loss = reduce_tensor(loss.data, cfg.world_size)
            losses_m.update(reduced_loss.item(), num_samples)

        update_task(progress_task, advance=cfg.world_size)
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()  # noqa

    lrl = [param_group['lr'] for param_group in optimizer.param_groups]
    metrics["loss"] = losses_m.avg
    metrics["lr"] = sum(lrl) / len(lrl)
    return metrics


def run_stage(
        stage: str,
        model: Union[DP, DDP],
        loss_fn: Callable,
        cfg: DLConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        gradient_scaler: Optional[NativeScaler],
        amp_autocast,
        meta: dict,
        distributed: bool,
        logger: logging.Logger,
) -> dict:
    logger.info(f"Training {stage}\n")
    try:
        with progress:
            epochs_num = cfg.stages[stage].total_epochs

            epoch_progress = task(f"[white]{stage}\n", total=epochs_num)
            train_progress = task(f"[blue]Train", total=len(train_dataloader) * cfg.world_size, start=False)
            val_progress = task(f"[red]Validation\n", total=len(val_dataloader) * cfg.world_size, start=False)

            for epoch in range(1, epochs_num + 1):
                if distributed and hasattr(train_dataloader.sampler, 'set_epoch'):  # noqa
                    train_dataloader.sampler.set_epoch(epoch - 1)  # noqa

                train_metrics = train_one_epoch(model, loss_fn, train_dataloader, optimizer, cfg,
                                                distributed=distributed,
                                                amp_autocast=amp_autocast,
                                                gradient_scaler=gradient_scaler,
                                                progress_task=train_progress,
                                                logger=logger)

                if distributed:
                    distribute_bn(model, cfg.world_size, cfg.training.reduce_bn)

                eval_metrics = val_epoch(model, loss_fn, val_dataloader, cfg,
                                         distributed=distributed,
                                         amp_autocast=amp_autocast,
                                         progress_task=val_progress)

                update_task(epoch_progress)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                track_metrics(cfg, stage, epoch, train_metrics, eval_metrics, meta, logger)

                if eval_metrics[cfg.evaluation.best_metric] > meta["stages"][stage]["best_metric"]:
                    meta["stages"][stage]["best_metric"] = eval_metrics[cfg.evaluation.best_metric]
                    meta["stages"][stage]["best_epoch"] = epoch

                    if cfg.local_rank == 0:
                        logger.info(f"Saving checkpoint for {stage}, epoch {epoch}\n")
                        save_checkpoint(model, meta["exp_dir"] / f"{stage}_best.pth")
                # end for
            # end with
    except KeyboardInterrupt:
        if cfg.local_rank == 0:
            logger.info("Training was interrupted. Saving last checkpoint\n")
            save_checkpoint(model, meta["exp_dir"] / f"{stage}_last.pth")
    return meta


def main(cfg: DLConfig):
    # init dist first before logging
    distributed = init_distributed(cfg)

    # prepare all the run stuff
    meta, logger = prepare_exp(cfg)

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = cfg.training.get('cudnn_benchmark', False)  # noqa

    # log some basic info
    logger.info(f'Distributed training: {distributed}\n')
    logger.info(f'Config:\n{cfg.pretty_text}\n')

    # init model
    model = object_from_dict(cfg.model)
    if cfg.model.use_preprocessing:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(
            cfg.model.params.encoder_name,
            cfg.model.params.encoder_weights
        )
    else:
        preprocessing_fn = None
    loss_fn = object_from_dict(cfg.loss)

    # put model on gpus, enable channels last layout if set
    model.cuda()
    if cfg.training.channels_last:
        model = model.to(memory_format=torch.channels_last)  # noqa

    # setup synchronized BatchNorm for distributed training
    if distributed:
        assert cfg.training.reduce_bn

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.local_rank == 0:
            logger.warning('Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                           'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.\n')

    if distributed:
        model = DDP(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank
        )
        model.find_unused_parameters = cfg.training.get('find_unused_parameters', False)
    else:
        model = DP(model, device_ids=[0])

    if cfg.training.load_from:
        logger.info(f'Load checkpoint from {cfg.training.load_from}\n')
        load_checkpoint(model, cfg.training.load_from, strict=False)

    if cfg.training.watch_model and cfg.local_rank == 0:
        wandb.watch(model, **cfg.training.watch_model)

    for stage_name, stage_cfg in cfg.stages.items():
        # set train datasets and loaders
        train_dataset = ChestDataset(**stage_cfg.train_dataset, preprocessing=preprocessing_fn)
        train_loader_cfg = get_loader_config(cfg, stage_name, train_dataset, distributed=distributed)
        train_dataloader = DataLoader(train_dataset, **train_loader_cfg)

        # set val datasets and loaders
        val_dataset = ChestDataset(**stage_cfg.val_dataset, preprocessing=preprocessing_fn)
        val_loader_cfg = get_loader_config(cfg, stage_name, val_dataset, kind="val", distributed=distributed)
        val_dataloader = DataLoader(val_dataset, **val_loader_cfg)

        # set optimizer and policy
        optimizer = object_from_dict(stage_cfg.optimizer, params=model.parameters())
        amp_autocast = nullcontext  # torch.cuda.amp.autocast
        lr_scheduler = object_from_dict(stage_cfg.lr_scheduler, optimizer=optimizer)
        gradient_scaler = None

        logger.info(f'Start running, host: {meta["host_name"]}, exp_dir: {meta["exp_dir"]}\n')
        meta = run_stage(stage=stage_name,
                         model=model,
                         loss_fn=loss_fn,
                         cfg=cfg,
                         train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader,
                         optimizer=optimizer,
                         amp_autocast=amp_autocast,
                         lr_scheduler=lr_scheduler,
                         gradient_scaler=gradient_scaler,
                         meta=meta,
                         distributed=distributed,
                         logger=logger)
        # end for
    log_artifacts(cfg, meta)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    config: DLConfig = load_config()
    main(config)
