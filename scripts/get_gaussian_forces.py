import hydra
from omegaconf import DictConfig

from diffenergy.gaussian_1d.inference import GaussianForces

@hydra.main(version_base=None, config_path='../configs')
def main(config:DictConfig):
    forces = GaussianForces(config)
    forces.get_forces()



if __name__ == "__main__":
    main()