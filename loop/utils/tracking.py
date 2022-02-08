from typing import Union

import wandb
from pathlib import Path


def log_artifact(name: str, artifact_path: Union[str, Path], type: str) -> None:
    if not isinstance(artifact_path, str):
        artifact_path = str(artifact_path)

    artifact = wandb.Artifact(name, type=type)
    artifact.add_file(artifact_path)
    wandb.log_artifact(artifact)
