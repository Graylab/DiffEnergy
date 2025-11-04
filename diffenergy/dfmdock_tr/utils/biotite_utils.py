from pathlib import Path
from typing import Iterable
from biotite.structure.io.general import load_structure, save_structure
from biotite.structure import AtomArray, get_chains
try:
    from biotite.structure import filter_peptide_backbone
except ImportError:
    from biotite.structure import filter_backbone as filter_peptide_backbone
import numpy as np
import torch
from diffenergy.dfmdock_tr.utils.geometry import axis_angle_to_matrix
from diffenergy.dfmdock_tr.utils.esm_utils import extract_coords_from_structure

def modify_aa_coords(x, rot, tr):
    center = x.mean(axis=0)
    rot = axis_angle_to_matrix(rot).squeeze().cpu().numpy()
    x = (x - center) @ rot.T + center 
    x = x + tr.cpu().numpy()
    return x

def get_chain_structure(orig:str|Path|AtomArray,chain:str,backbone_only:bool=False):
    if isinstance(orig,AtomArray):
        orig_structure = orig
    else:
        orig_structure:AtomArray = load_structure(orig)
        
    all_chains = get_chains(orig_structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input.")
    elif chain not in all_chains:
        raise ValueError(f"Cannot offset chain {chain}; not in input!")
    
    if backbone_only:
        bbmask = filter_peptide_backbone(orig_structure)
        orig_structure = orig_structure[bbmask]

    #we can extract just the ligands, modify the structure, then assign it back using this boolean mask
    lig_filter = orig_structure.chain_id == chain #get boolean mask
    lig_structure = orig_structure[lig_filter]
    
    return lig_structure

#same as load_coords from esm_utils, but can take an already loaded atomarray as input to save disk read
def get_chain_coords(orig:str|Path|AtomArray,chain:str,backbone_only:bool=True):
    return extract_coords_from_structure(get_chain_structure(orig,chain,backbone_only=backbone_only))


# load pdb with **all** atoms, not just backbone atoms, and offset specified chain. Defaults to B cause that's the default ligand chain
def get_offset_pdb(
        orig:str|Path|AtomArray,
        offset_tr:None|torch.Tensor,
        offset_rot:None|torch.Tensor,
        offset_chain="B"
        ):
    if isinstance(orig,AtomArray):
        orig_structure = orig
    else:
        orig_structure:AtomArray = load_structure(orig)
        
    all_chains = get_chains(orig_structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input.")
    elif offset_chain not in all_chains:
        raise ValueError(f"Cannot offset chain {offset_chain}; not in input!")

    #we can extract just the ligands, modify the structure, then assign it back using this boolean mask
    lig_filter = orig_structure.chain_id == offset_chain #get boolean mask
    lig_structure = orig_structure[lig_filter]

    offset_rot = offset_rot if offset_rot is not None else torch.zeros((3,),dtype=float)
    offset_tr = offset_tr if offset_tr is not None else torch.zeros((3,),dtype=float)

    #actually since we're modifying coord in-place I don't know if we need to re-assign but it probably makes a copy so safer than sorry
    lig_structure.coord = modify_aa_coords(lig_structure.coord,offset_rot,offset_tr) 
    
    orig_structure[lig_filter] = lig_structure
    
    return orig_structure