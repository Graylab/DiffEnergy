

import hydra
from omegaconf import DictConfig

from diffenergy.dfmdock_tr.inference import DFMDockLikelihood


@hydra.main(version_base=None, config_path="../configs/likelihoodv3")
def main(config:DictConfig):
    likelihood = DFMDockLikelihood(config)
    likelihood.compute_likelihoods()


if __name__ == "__main__":
    main()