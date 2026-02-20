import os
import hydra
from hydra.core.config_store import ConfigStore
from src.config import MyConfig
# from src.experiments import Experiment

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)

@hydra.main(config_path="src/conf", config_name="gazelle_train", version_base="1.1")
def main(cfg: MyConfig) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.experiment.cuda)

    from src.experiments import Experiment
    experiment = Experiment(cfg)
    experiment.setup()
    experiment.run()

if __name__ == "__main__":
    # print(f"I am here: {os.getcwd()}")
    main()
