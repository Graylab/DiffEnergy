
import hydra
from omegaconf import DictConfig

from diffenergy.dfmdock_tr.inference import DFMDockForces

@hydra.main(version_base=None, config_path='../configs/likelihoodv3')
def main(config:DictConfig):
    forces = DFMDockForces(config)
    forces.get_forces()            


if __name__ == "__main__":
    main()