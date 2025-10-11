### inspired from Leeshin Chu's dips_dataset.py
import contextlib
from typing import TypedDict
import warnings
import torch
import torch.nn.functional as F
import os.path as path
from torch.utils import data
from diffenergy.dfmdock_tr.utils.esm_utils import load_coords # Utils file from ESM https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py
from diffenergy.dfmdock_tr.utils.pdb import save_PDB, place_fourth_atom
from diffenergy.dfmdock_tr.utils import residue_constants
from diffenergy.dfmdock_tr.esm_model import ESMLanguageModel 
from esm.data import Alphabet

class DockedDatum(TypedDict):
    id: str
    pdb_file: str
    rec_seq: str
    lig_seq: str
    rec_x: torch.Tensor
    lig_x: torch.Tensor
    rec_pos: torch.Tensor
    lig_pos: torch.Tensor
    position_matrix: torch.Tensor


#----------------------------------------------------------------------------
class PDBImporter:
    def __init__(
        self, 
        esm_model:ESMLanguageModel,
        esm_alphabet:Alphabet,
    ):
        # Path to the data directory 
        self.esm_model = esm_model
        self.batch_converter = esm_alphabet.get_batch_converter()

    def get_pdb(self,pdb_file:str,id:str,out_pdb: bool = False, suppress_warnings: bool = True)->DockedDatum:

        # Get sequences and coords from files  
        ctx = warnings.catch_warnings(action="ignore") if suppress_warnings else contextlib.nullcontext()
        with ctx:
            rec_pos, rec_seq = load_coords(pdb_file,"A")
            lig_pos, lig_seq = load_coords(pdb_file,"B")


        # Convert coords to torch tensor
        rec_pos = self.convert_to_torch_tensor(rec_pos)
        lig_pos = self.convert_to_torch_tensor(lig_pos)
        rec_pos = rec_pos[:len(rec_seq), :, :]
        lig_pos = lig_pos[:len(lig_seq), :, :]

        # Get esm tokenize
        rec_x = self.get_tokens(rec_seq).unsqueeze(0)
        lig_x = self.get_tokens(lig_seq).unsqueeze(0)

        if out_pdb:
            test_coords = torch.cat([rec_pos, lig_pos], dim=0)
            test_coords = self.get_full_coords(test_coords)
            save_PDB('test.pdb', test_coords, rec_seq+lig_seq, len(rec_seq)-1)

        # get esm embedding
        rec_x = self.esm_model(rec_x).squeeze(0) 
        lig_x = self.esm_model(lig_x).squeeze(0)  
    
        # get one-hot encodings from openfold
        rec_onehot = torch.from_numpy(residue_constants.sequence_to_onehot(
        sequence=rec_seq,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
        )).float()

        lig_onehot = torch.from_numpy(residue_constants.sequence_to_onehot(
        sequence=lig_seq,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
        )).float() 

        # Concat esm embedding and one-hot encodings
        rec_x = torch.cat([rec_x, rec_onehot], dim=-1)   
        lig_x = torch.cat([lig_x, lig_onehot], dim=-1)    

        # get res_id and asym_id
        n = rec_x.size(0) + lig_x.size(0)
        res_id = torch.arange(n).long()
        asym_id = torch.zeros(n).long()
        asym_id[rec_x.size(0):] = 1

        # positional embeddings
        position_matrix = self.relpos(res_id, asym_id)

        # Output
        output:DockedDatum = {
            'id': id,
            'pdb_file': pdb_file,
            'rec_seq': rec_seq,
            'lig_seq': lig_seq,
            'rec_x': rec_x,
            'lig_x': lig_x,
            'rec_pos': rec_pos,
            'lig_pos': lig_pos,
            'position_matrix': position_matrix,
        }
        
        return output

    @classmethod
    def relpos(cls, res_id, asym_id, use_chain_relative=True):
        max_relative_idx = 32
        pos = res_id
        asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(
            offset + max_relative_idx, 0, 2 * max_relative_idx
        )

        rel_feats = []
        if use_chain_relative:
            final_offset = torch.where(
                asym_id_same, 
                clipped_offset,
                (2 * max_relative_idx + 1) * 
                torch.ones_like(clipped_offset)
            )

            boundaries = torch.arange(
                start=0, end=2 * max_relative_idx + 2
            )
            rel_pos = cls.one_hot(
                final_offset,
                boundaries,
            )

            rel_feats.append(rel_pos)

        else:
            boundaries = torch.arange(
                start=0, end=2 * max_relative_idx + 1
            )
            rel_pos = cls.one_hot(
                clipped_offset, boundaries,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).float()

        return rel_feat

    @classmethod
    def one_hot(cls, x, v_bins):
        reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
        diffs = x[..., None] - reshaped_bins
        am = torch.argmin(torch.abs(diffs), dim=-1)
        
        return F.one_hot(am, num_classes=len(v_bins)).float()

    def get_tokens(self, seq_prim):
        
        # Use ESM-1b format.
        # The length of tokens is:
        # L (sequence length) + 2 (start and end tokens)
        
        seq = [
            ("seq", seq_prim)
        ]
        *_, tokens = self.batch_converter(seq)

        return tokens.squeeze_(0)

    def convert_to_torch_tensor(self, atom_coords):
        
        # Convert atom_coords to torch tensor.
        
        n_coords = torch.Tensor(atom_coords[:,0,:])
        ca_coords = torch.Tensor(atom_coords[:,1,:])
        c_coords = torch.Tensor(atom_coords[:,2,:])
        coords = torch.stack([n_coords, ca_coords, c_coords], dim=1)
        return coords


    def get_full_coords(self, coords):
        
        n_coords, ca_coords, c_coords = coords[:,0,:], coords[:,1,:], coords[:,2,:]    
        
        #get full coords
        
        cb_coords = place_fourth_atom(c_coords, n_coords, ca_coords,
                                        torch.tensor(1.522),
                                        torch.tensor(1.927),
                                        torch.tensor(-2.143))
        o_coords = place_fourth_atom(torch.roll(n_coords, -1, 0),
                                        ca_coords, c_coords,
                                        torch.tensor(1.231),
                                        torch.tensor(2.108),
                                        torch.tensor(-3.142))
        full_coords = torch.stack(
            [n_coords, ca_coords, c_coords, o_coords, cb_coords], dim=1)
        
        return full_coords
    
class DockingDataset(data.Dataset[DockedDatum]):
    def __init__(
        self, 
        data_dir: str,
        data_list: str,
        importer: PDBImporter,
        out_pdb: bool = False,
    ):
        # Path to the data directory 
        self.data_dir = data_dir
        self.data_list = data_list
        self.out_pdb = out_pdb
        with open(self.data_list, 'r') as f:
            lines = f.readlines()
        self.file_list = [line.strip() for line in lines] 
        self.importer = importer

    def __getitem__(self, idx) -> DockedDatum:
        pdb_file = path.join(self.data_dir, self.file_list[idx])
        _id = self.file_list[idx]
        return self.importer.get_pdb(pdb_file,_id,out_pdb=self.out_pdb)

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    # data_dir = "/scratch4/jgray21/ssarma4/pdbs"
    data_dir = "/home/dxu39/scr4_jgray21/dxu39/projects/diffenergy/dfmdock_perturb_tr_likelihood/dfmdock_inference/results/trjs/db5_test_DFMDock_model_0_0.5_120_samples_40_steps_dips/splits"
    # data_list = "/scratch4/jgray21/ssarma4/pdbs/filenames.txt"
    data_list = "/home/dxu39/scr4_jgray21/dxu39/projects/diffenergy/dfmdock_perturb_tr_likelihood/dfmdock_inference/results/trjs/dfmdock_perturb_tr_likelihood_db5_test_trj.txt"
    dataset = DockingDataset(
        data_dir=data_dir, 
        data_list=data_list, 
        out_pdb=False,
    )
    from IPython import embed; embed()
