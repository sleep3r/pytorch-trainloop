import git
import json
import logging
from pathlib import Path
from datetime import datetime
from getpass import getuser
from socket import gethostname
from contextlib import nullcontext
from typing import Optional, Union, Callable
from collections import defaultdict

import numpy as np
import wandb
import torch
import torch.distributed as dist
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from rich.progress import TaskID

from config import load_config, DLConfig, object_from_dict
from dataset import ChestDataset, OrderedDistributedSampler
from checkpoint import load_checkpoint, save_checkpoint
from utils.clip_grad import dispatch_clip_grad
from utils.distributed import init_distributed, distribute_bn, reduce_tensor
from utils.env import collect_env
from utils.io import read_duplicates, read_norma_ids
from utils.logging import get_logger, status
from utils.metrics import calc_metrics, AverageMeter
from utils.path import mkdir_or_exist
from utils.cuda import NativeScaler
from utils.progress import progress, task, refresh_task, update_task, remove_task
from utils.tracking import log_artifact
from utils.training import set_random_seed


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

    # setup w&b tracking
    meta = setup_tracking(cfg, meta)

    # setup eval metrics tracking
    meta = define_metrics(cfg, meta, logger=logger)
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


def env_collect(cfg: DLConfig, meta: dict, logger: logging.Logger) -> dict:
    """Collects environment information."""
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"

    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    repo = git.Repo(search_parent_directories=True)
    meta["sha"] = repo.head.object.hexsha
    meta["host_name"] = f"{getuser()}@{gethostname()}"
    return meta


def determine_exp(cfg: DLConfig, meta: dict, logger: logging.Logger) -> dict:
    """Sets seed and experiment name."""
    if cfg.training.seed is not None:
        logger.info(f"Set random seed to {cfg.training.seed}, deterministic: {cfg.training.deterministic}\n")
        set_random_seed(
            cfg.training.seed,
            precision=cfg.training.precision,
            deterministic=cfg.training.deterministic,
        )

    meta["seed"] = cfg.training.seed
    meta["exp_name"] = cfg.training.exp_name
    return meta


def define_metrics(cfg: DLConfig, meta: dict, logger: logging.Logger) -> dict:
    meta["crossval"] = defaultdict(dict)

    for cv_step in range(1, cfg.cross_validation.n_splits + 1):
        meta["crossval"][cv_step]["train_loss"] = []
        meta["crossval"][cv_step]["val_loss"] = []

        meta["crossval"][cv_step]["lr"] = []

        meta["crossval"][cv_step]["best_metric"] = 0
        meta["crossval"][cv_step]["best_epoch"] = 0

        if cfg.local_rank == 0:
            wandb.define_metric(f"CV {cv_step}/*", step_metric=f"CV {cv_step}/step")
    return meta


def setup_tracking(cfg: DLConfig, meta: dict) -> dict:
    """Sets up mlflow tracking."""
    if cfg.local_rank == 0:
        run = wandb.init(
            project="chest_uncertainty",
            name=meta["run_name"],
            config=cfg.to_dict(),
        )
        meta["exp_url"] = run.get_url()
    return meta


def track_metrics(
        cfg: DLConfig,
        cv_step: int,
        epoch: int,
        train_metrics: dict,
        eval_metrics: dict,
        meta: dict,
        logger: logging.Logger,
) -> None:
    if cfg.local_rank == 0:
        dash_line = "-" * 60 + "\n"

        meta["crossval"][cv_step]["train_loss"].append(train_metrics["loss"])
        meta["crossval"][cv_step]["val_loss"].append(eval_metrics["loss"])

        # loss
        loss_info = f'[Epoch]: {epoch}\nTrain loss: {train_metrics["loss"]}\nVal loss: {eval_metrics["loss"]}\n'
        wandb.log({f"CV {cv_step}/train_loss": train_metrics["loss"], f"CV {cv_step}/step": epoch})
        wandb.log({f"CV {cv_step}/val_loss": eval_metrics["loss"], f"CV {cv_step}/step": epoch})

        # metrics
        metrics_info = "\n".join([f"{k}: {v}" for k, v in eval_metrics.items() if k != "loss"])
        for metric in cfg.evaluation.metrics:
            wandb.log({f"CV {cv_step}/{metric}": eval_metrics[metric], f"CV {cv_step}/step": epoch})

        # lr
        wandb.log({f"CV {cv_step}/lr": train_metrics["lr"], f"CV {cv_step}/step": epoch})

        logger.info(dash_line + loss_info + "\n" + metrics_info + "\n" + dash_line)


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
        dataset: ChestDataset,
        distributed: bool,
        kind: str = "train",
) -> dict:
    """Builds loader config."""
    collate_fn = object_from_dict(cfg.loader.get("collate_fn", {}))

    loader_cfg = {
        **cfg.loader,
        **dict(collate_fn=collate_fn),
    }
    # TRAIN
    if kind == "train":
        if distributed:
            loader_cfg["sampler"] = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=True,
                num_replicas=cfg.world_size,
                rank=cfg.local_rank,  # noqa
            )

    # VAL
    elif kind == "val":
        if distributed:
            loader_cfg["sampler"] = OrderedDistributedSampler(
                dataset,
                num_replicas=cfg.world_size,
                rank=cfg.local_rank
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
            data_batch["label"] = data_batch["label"].cuda()

            if cfg.training.channels_last:
                data_batch["image"] = data_batch["image"].contiguous(memory_format=torch.channels_last)  # noqa

            with amp_autocast():
                outputs = model(data_batch["image"])

                if cfg.model.params.aux_params is not None:
                    # with clf head
                    seg_loss = loss_fn(outputs[0], data_batch["mask"].float())
                    clf_loss = torch.nn.BCELoss()(outputs[1][:, 1], data_batch["label"])
                    loss = seg_loss + clf_loss
                    outputs = outputs[0]
                else:
                    loss = loss_fn(outputs, data_batch["mask"].float())

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
        logger: logging.Logger = None,
):
    metrics = {}
    losses_m = AverageMeter()

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order

    model.train()

    last_idx = len(loader) - 1
    refresh_task(progress_task)
    for batch_idx, data_batch in enumerate(loader):
        last_batch = batch_idx == last_idx

        data_batch["image"] = data_batch["image"].cuda()
        data_batch["mask"] = data_batch["mask"].cuda()
        data_batch["label"] = data_batch["label"].cuda()

        if cfg.training.channels_last:
            data_batch["image"] = data_batch["image"].contiguous(memory_format=torch.channels_last)  # noqa

        with amp_autocast():
            outputs = model(data_batch["image"])

            if cfg.model.params.aux_params is not None:
                # with clf head
                seg_loss = loss_fn(outputs[0], data_batch["mask"].float())
                clf_loss = torch.nn.BCELoss()(outputs[1][:, 1], data_batch["label"])
                loss = seg_loss + clf_loss
            else:
                loss = loss_fn(outputs, data_batch["mask"].float())

            num_samples = data_batch["image"].size(0)

        if not distributed:
            losses_m.update(loss.item(), num_samples)

        optimizer.zero_grad()
        if gradient_scaler is not None:
            gradient_scaler(
                loss,
                optimizer,
                clip_grad=cfg.training.clip_grad,
                clip_mode=cfg.training.clip_mode,
                parameters=model.parameters(),
                create_graph=second_order,
            )
        else:
            loss.backward(create_graph=second_order)

            if cfg.training.clip_grad is not None:
                dispatch_clip_grad(
                    model.parameters(),
                    value=cfg.training.clip_grad,
                    mode=cfg.training.clip_mode,
                )

            optimizer.step()

        torch.cuda.synchronize()

        if last_batch and distributed:
            reduced_loss = reduce_tensor(loss.data, cfg.world_size)
            losses_m.update(reduced_loss.item(), num_samples)

        update_task(progress_task, advance=cfg.world_size)
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()  # noqa

    lrl = [param_group["lr"] for param_group in optimizer.param_groups]
    metrics["loss"] = losses_m.avg
    metrics["lr"] = sum(lrl) / len(lrl)
    return metrics


def run_cv_step(
        cv_step: int,
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
    """
    Runs training and validation for one crossval step.

    Args:
        cv_step: crossval step number;
        model: model to train;
        loss_fn: loss function;
        cfg: config;
        train_dataloader: training dataloader;
        val_dataloader: validation dataloader;
        optimizer: optimizer;
        lr_scheduler: learning rate scheduler;
        gradient_scaler: gradient scaler;
        amp_autocast: mixed precision function;
        meta: meta data;
        distributed: whether to use distributed training;
        logger: logger.
    Returns:
        dict: updated meta training dictionary.
    """
    with progress:
        epochs_num = cfg.training.total_epochs

        epoch_progress = task(f"[white]CV {cv_step}\n", total=epochs_num)
        train_progress = task(f"[blue]Train", total=len(train_dataloader) * cfg.world_size, start=False)
        val_progress = task(f"[red]Validation\n", total=len(val_dataloader) * cfg.world_size, start=False)

        for epoch in range(1, epochs_num + 1):
            if distributed and hasattr(train_dataloader.sampler, "set_epoch"):  # noqa
                train_dataloader.sampler.set_epoch(epoch - 1)  # noqa

            train_metrics = train_one_epoch(
                model,
                loss_fn,
                train_dataloader,
                optimizer,
                cfg,
                distributed=distributed,
                amp_autocast=amp_autocast,
                gradient_scaler=gradient_scaler,
                progress_task=train_progress,
                logger=logger,
            )

            if distributed:
                distribute_bn(model, cfg.world_size, cfg.training.reduce_bn)

            eval_metrics = val_epoch(
                model,
                loss_fn,
                val_dataloader,
                cfg,
                distributed=distributed,
                amp_autocast=amp_autocast,
                progress_task=val_progress,
            )

            update_task(epoch_progress)

            if lr_scheduler is not None:
                lr_scheduler.step()

            track_metrics(cfg, cv_step, epoch, train_metrics, eval_metrics, meta, logger)

            if eval_metrics[cfg.evaluation.best_metric] > meta["crossval"][cv_step]["best_metric"]:
                meta["crossval"][cv_step]["best_metric"] = eval_metrics[cfg.evaluation.best_metric]
                meta["crossval"][cv_step]["best_epoch"] = epoch

                if cfg.local_rank == 0:
                    logger.info(f"Saving checkpoint for crossval step {cv_step}, epoch {epoch}\n")
                    save_checkpoint(model, meta["exp_dir"] / f"CV{cv_step}_best.pth")
            # end for
        remove_task(epoch_progress)
        remove_task(train_progress)
        remove_task(val_progress)
        # end with
    return meta


def main(cfg: DLConfig):
    # init dist first before logging
    distributed = init_distributed(cfg)

    # prepare all the run stuff
    meta, logger = prepare_exp(cfg)

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = cfg.training.get("cudnn_benchmark", False)  # noqa

    # log some basic info
    logger.info(f"Distributed training: {distributed}\n")
    logger.info(f"Config:\n{cfg.pretty_text}\n")

    # init model
    model = object_from_dict(cfg.model)

    loss_fn = object_from_dict(cfg.loss)

    # put model on gpus, enable channels last layout if set
    model.cuda()  # noqa
    if cfg.training.channels_last:
        model = model.to(memory_format=torch.channels_last)  # noqa

    # setup synchronized BatchNorm for distributed training
    if distributed and cfg.training.reduce_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if cfg.local_rank == 0:
            logger.warning(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.\n"
            )

    if distributed:
        model = DDP(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=cfg.training.get("find_unused_parameters", False)  # noqa
        )
        cfg.training.get("find_unused_parameters", False)
    else:
        model = DP(model, device_ids=[0])

    # saving base model weights for further reinitialization during CV
    save_checkpoint(model, meta["exp_dir"] / "base_model.pth")

    # set optimizer and policy
    optimizer = object_from_dict(cfg.optimizer, params=model.parameters())
    amp_autocast = nullcontext  # torch.cuda.amp.autocast
    lr_scheduler = object_from_dict(cfg.lr_scheduler, optimizer=optimizer)
    gradient_scaler = None

    # here we will perform cross validation
    # we will do groups split on the known data ids, using duplicates data
    duplicates = read_duplicates(cfg.cross_validation.duplicates_path)
    groups = [*duplicates.values()]
    group_cv = GroupKFold(n_splits=cfg.cross_validation.n_splits)
    pneumonia_ids = np.array([*duplicates.keys()], dtype=int)
    pneumonia_split = group_cv.split(pneumonia_ids, None, groups)

    # no need for such grouping of norma images
    norma_ids = np.array(read_norma_ids(cfg.training.images_path), dtype=int)
    cv = KFold(n_splits=cfg.cross_validation.n_splits, shuffle=True)
    norma_split = cv.split(norma_ids)

    logger.info(f'Start running, host: {meta["host_name"]}, exp_dir: {meta["exp_dir"]}\n')
    for cv_step in range(cfg.cross_validation.n_splits):
        logger.info(f"Cross validation step: {cv_step + 1}\n")
        try:
            # reset seed for each cross validation step
            cfg.training.seed += cfg.training.fold_seed_step
            set_random_seed(cfg.training.seed)

            if cfg.training.load_from:
                logger.info(f"Load checkpoint from {cfg.training.load_from}\n")
                load_checkpoint(model, cfg.training.load_from, strict=False)
            else:
                logger.info(f"Model has been reinitialized\n")
                load_checkpoint(model, meta["exp_dir"] / "base_model.pth")
                model.module.initialize()

            # reading images ids for current cv split
            pneumonia_train, pneumonia_val = next(pneumonia_split)
            norma_train, norma_val = next(norma_split)

            # set train datasets and loaders
            train_subsamples = (pneumonia_ids[pneumonia_train], norma_ids[norma_train])
            train_dataset = ChestDataset(**cfg.train_dataset, subsamples=train_subsamples)
            train_loader_cfg = get_loader_config(cfg, train_dataset, kind="train", distributed=distributed)
            train_dataloader = DataLoader(train_dataset, **train_loader_cfg)

            # set val datasets and loaders
            val_subsamples = (pneumonia_ids[pneumonia_val], norma_ids[norma_val])
            val_dataset = ChestDataset(**cfg.val_dataset, subsamples=val_subsamples)
            val_loader_cfg = get_loader_config(cfg, val_dataset, kind="val", distributed=distributed)
            val_dataloader = DataLoader(val_dataset, **val_loader_cfg)

            meta = run_cv_step(
                cv_step=cv_step + 1,
                model=model,
                loss_fn=loss_fn,  # noqa
                cfg=cfg,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,  # noqa
                amp_autocast=amp_autocast,
                lr_scheduler=lr_scheduler,
                gradient_scaler=gradient_scaler,
                meta=meta,
                distributed=distributed,
                logger=logger,
            )
        except KeyboardInterrupt:
            if cfg.local_rank == 0:
                logger.info("Training was interrupted. Saving last checkpoint\n")
                save_checkpoint(model, meta["exp_dir"] / f"CV{cv_step}_last.pth")
            break

    log_artifacts(cfg, meta)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    config: DLConfig = load_config()
    main(config)
