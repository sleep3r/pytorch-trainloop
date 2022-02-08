from typing import Optional

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TaskID

from utils.distributed import get_dist_info

text_column = TextColumn("{task.description}")
bar_column = BarColumn(bar_width=None, style="bar.back")
time_column = TimeElapsedColumn()

# main progress bar context manager
progress = Progress(
    text_column,
    bar_column,
    "[progress.percentage]{task.completed}/{task.total}",
    time_column,
    transient=True
)


def task(description: str, total: int, start: bool = True) -> Optional[TaskID]:
    """
    Creates a new progress-bar task with the given description and total steps.

    Args:
        description: text description of the task;
        total: total number of steps;
        start: whether to start the task immediately.
    Returns:
        task_id: task ID for local_rank = 0.
    """
    rank, world_size = get_dist_info()
    if rank == 0:
        return progress.add_task(description=description, total=total, start=start)
    else:
        return None


def refresh_task(task_id: Optional[TaskID]) -> None:
    """
    Resets the progress bar.

    Args:
        task_id: task ID for local_rank = 0.
    """
    if task_id is not None:
        progress.start_task(task_id)
        progress.reset(task_id)


def update_task(task_id: Optional[TaskID], advance: int = 1) -> None:
    """
    Updates the progress bar with the given advance amount.

    Args:
        task_id: task ID for local_rank = 0.
        advance: number of steps to advance.
    """
    if task_id is not None:
        progress.start_task(task_id)
        progress.update(task_id, advance=advance)


def reset_task(task_id: Optional[TaskID]) -> None:
    """
    Stops and resets the progress bar.

    Args:
        task_id: task ID for local_rank = 0.
    """
    if task_id is not None:
        progress.stop_task(task_id)
        progress.reset(task_id)
