from pytorch_lightning.callbacks import RichProgressBar

class CustomRichProgressBar(RichProgressBar):
    def __init__(self, refresh_rate: int = 1, leave: bool = False):
        super().__init__(refresh_rate=refresh_rate, leave=leave)

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        # Remove both 'v_num' and 'version'
        items.pop("v_num", None)
        items.pop("version", None)
        return items
