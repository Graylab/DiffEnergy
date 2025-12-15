

import hydra
from omegaconf import DictConfig

from diffenergy.gaussian_1d.inference import GaussianLikelihood


@hydra.main(version_base=None, config_path="../configs")
def main(config:DictConfig):
    likelihood = GaussianLikelihood(config)
    likelihood.compute_likelihoods()


if __name__ == "__main__":
    main()