import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def my_app(cfg: DictConfig) -> None:
    print("This is a test. If you see this, Hydra is working.")
    log.info("This is a test log message.")
    # The real output dir is determined by hydra's runtime,
    # but we can see the configured pattern.
    log.info(f"Hydra run.dir pattern: {cfg.run.dir}")

if __name__ == "__main__":
    my_app()
