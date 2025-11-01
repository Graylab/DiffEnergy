from pathlib import Path
from biotite.structure.io.general import load_structure, save_structure
from biotite.structure import AtomArray, get_chains
import numpy as np
import torch
from diffenergy.dfmdock_tr.utils.geometry import axis_angle_to_matrix

def modify_aa_coords(x, rot, tr):
    center = x.mean(axis=0)
    rot = axis_angle_to_matrix(rot).squeeze().cpu().numpy()
    x = (x - center) @ rot.T + center 
    x = x + tr.cpu().numpy()
    return x

# load pdb with **all** atoms, not just backbone atoms, and offset specified chain. Defaults to B cause that's the default ligand chain
def get_offset_pdb(
        orig:str|Path,
        offset_tr:None|torch.Tensor,
        offset_rot:None|torch.Tensor,
        offset_chain="B"
        ):
    orig_structure:AtomArray = load_structure(orig)
    all_chains = get_chains(orig_structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    elif offset_chain not in all_chains:
        raise ValueError(f"Cannot offset chain {offset_chain}; not in file!")

    #we can extract just the ligands, modify the structure, then assign it back using this boolean mask
    lig_filter = orig_structure.chain_id == offset_chain #get boolean mask
    lig_structure = orig_structure[lig_filter]

    offset_rot = offset_rot if offset_rot is not None else torch.zeros((3,),dtype=float)
    offset_tr = offset_tr if offset_tr is not None else torch.zeros((3,),dtype=float)

    #actually since we're modifying coord in-place I don't know if we need to re-assign but it probably makes a copy so safer than sorry
    lig_structure.coord = modify_aa_coords(lig_structure.coord,offset_rot,offset_tr) 
    
    orig_structure[lig_filter] = lig_structure
    
    return orig_structure