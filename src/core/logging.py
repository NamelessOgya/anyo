import logging
import time
import json
from pathlib import Path
from typing import Dict
from contextlib import ContextDecorator

from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


def setup_logging(log_file: Path):
    """Sets up basic logging to console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


class TensorBoardLogger:
    """Manages TensorBoard SummaryWriters."""

    def __init__(self, log_dir: Path):
        self.writers: Dict[str, SummaryWriter] = {}
        self.log_dir = log_dir

    def get_writer(self, name: str) -> SummaryWriter:
        """Returns a SummaryWriter for a given name, creating it if it doesn't exist."""
        if name not in self.writers:
            writer_path = self.log_dir / name
            writer_path.mkdir(parents=True, exist_ok=True)
            self.writers[name] = SummaryWriter(log_dir=str(writer_path))
        return self.writers[name]

    def close_all(self):
        """Closes all active SummaryWriters."""
        for writer in self.writers.values():
            writer.close()


class time_block(ContextDecorator):
    """
    Context manager to measure and log execution time of a block of code.
    Times are stored in a dictionary and saved to a JSON file.
    """

    _times: Dict[str, float] = {}
    _output_path: Path = Path(
        "metrics/time.json"
    )  # Default, will be updated by main script

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        if self.start_time is not None:
            duration = end_time - self.start_time
            time_block._times[self.name] = duration
            log.info(f"Time block '{self.name}' executed in {duration:.4f} seconds.")
            self.save_times()

    @classmethod
    def set_output_path(cls, path: Path):
        """Sets the output path for the time.json file."""
        cls._output_path = path
        cls._output_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def save_times(cls):
        """Saves the accumulated times to a JSON file."""
        if not cls._output_path.parent.exists():
            cls._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cls._output_path, "w") as f:
            json.dump(cls._times, f, indent=4)
        log.info(f"Execution times saved to {cls._output_path}")

    @classmethod
    def get_times(cls) -> Dict[str, float]:
        """Returns the accumulated times."""
        return cls._times

    @classmethod
    def clear_times(cls):
        """Clears the accumulated times."""
        cls._times = {}


# Example usage (for testing/demonstration)
if __name__ == "__main__":
    # Setup logging to console and a dummy file
    test_log_file = Path("test_log.log")
    setup_logging(test_log_file)
    log.info("Logging setup complete.")

    # Setup TensorBoard
    tb_logger = TensorBoardLogger(Path("tb_logs"))
    writer = tb_logger.get_writer("test_run")
    writer.add_scalar("test/loss", 0.5, 1)
    writer.add_scalar("test/loss", 0.3, 2)
    tb_logger.close_all()
    log.info("TensorBoard logging complete.")

    # Setup time_block output path
    time_block.set_output_path(Path("test_metrics/time.json"))

    # Use time_block
    with time_block("task_a"):
        time.sleep(0.1)

    @time_block("task_b")
    def my_function():
        time.sleep(0.2)

    my_function()

    with time_block("task_c"):
        time.sleep(0.05)
        with time_block("sub_task_c1"):
            time.sleep(0.02)

    log.info(f"All recorded times: {time_block.get_times()}")
    time_block.save_times()
    time_block.clear_times()
    log.info(f"Times after clearing: {time_block.get_times()}")

    # Clean up test files
    test_log_file.unlink(missing_ok=True)
    Path("tb_logs/test_run").rmdir()
    Path("tb_logs").rmdir()
    Path("test_metrics/time.json").unlink(missing_ok=True)
    Path("test_metrics").rmdir()
