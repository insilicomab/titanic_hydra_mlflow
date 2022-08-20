from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.feature.ohe_col)


if __name__ == "__main__":
    my_app()
