import time

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate
from torch_mist.utils.logging.logger.wandb import WandbLogger

from datasets.utils import NormalizedDataset


@hydra.main(
    config_path="config", config_name="config.yaml", version_base="1.1"
)
def parse(conf: DictConfig):
    logger = instantiate(conf.logger, _convert_='all')
    if hasattr(logger, "add_config"):
        logger.add_config(OmegaConf.to_container(conf, resolve=True))

    data = instantiate(conf.data, _convert_='all')

    if conf.normalize_data:
        data = NormalizedDataset(data)

    method = instantiate(conf.method,
        _partial_=True
    )


    mi_estimate, log = method(
        logger=logger,
        data=data
    )

    ## logger.save_log()
    print(f'Mutual Information: {mi_estimate} nats')


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    try:
        parse()
    except Exception as e:
        print(e)





