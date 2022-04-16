import json
from typing import Union

import wandb
from pathlib import Path

from utils.path import mkdir_or_exist


def log_artifact(name: str, artifact_path: Union[str, Path], type: str) -> None:
    if not isinstance(artifact_path, str):
        artifact_path = str(artifact_path)

    artifact = wandb.Artifact(name, type=type)
    artifact.add_file(artifact_path)
    wandb.log_artifact(artifact)


def save_splits(train_subsamples, val_subsamples, cv_step, exp_dir: Path):
    mkdir_or_exist(str(exp_dir / "CV_splits"))

    with open(exp_dir / "CV_splits" / f"{cv_step + 1}.json", "w") as f:
        splits = {
            "train": {
                "pneumonia": train_subsamples[0].tolist(),
                "norma": train_subsamples[1].tolist()
            },
            "val": {
                "pneumonia": val_subsamples[0].tolist(),
                "norma": val_subsamples[1].tolist()
            }
        }
        json.dump(splits, f, indent=2)
