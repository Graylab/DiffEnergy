

import hydra
from omegaconf import DictConfig

from diffenergy.dfmdock_tr.inference import DFMDockSampler

@hydra.main(version_base=None, config_path="../configs")
def main(config:DictConfig):
    sampler = DFMDockSampler(config)
    sampler.sample()

if __name__ == "__main__":
    main()