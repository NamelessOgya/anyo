from pytorch_lightning.callbacks import RichProgressBar
from rich.progress import TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, Progress

class CustomRichProgressBar(RichProgressBar):
    def __init__(self, refresh_rate: int = 1, leave: bool = False):
        super().__init__(refresh_rate=refresh_rate, leave=leave)

    def configure_columns(self, trainer) -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            self.metrics_column,
        ]

    def get_metrics(self, trainer, model):
        # Don't show the main progress bar metrics when val/test progress bar is active.
        if trainer.progress_bar_callback.is_running and not trainer.sanity_checking:
            return {}
        
        items = super().get_metrics(trainer, model)
        # Remove both 'v_num' and 'version'
        items.pop("v_num", None)
        items.pop("version", None)
        return items
