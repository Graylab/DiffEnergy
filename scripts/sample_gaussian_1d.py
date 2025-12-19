

import hydra
from omegaconf import DictConfig

from diffenergy.gaussian_1d.inference import GaussianSampler

@hydra.main(version_base=None, config_path="../configs")
def main(config:DictConfig):
    sampler = GaussianSampler(config)
    sampler.sample()


if __name__ == "__main__":
    main()