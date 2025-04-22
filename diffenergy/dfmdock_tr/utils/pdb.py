import re
import torch
import numpy as np
from os.path import splitext, basename
from Bio import SeqIO
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.ResidueDepth import residue_depth, get_surface
from Bio.PDB.SASA import ShrakeRupley


_aa_dict = {
    'A': '0',
    'C': '1',
    'D': '2',
    'E': '3',
    'F': '4',
    'G': '5',
    'H': '6',
    'I': '7',
    'K': '8',
    'L': '9',
    'M': '10',
    'N': '11',
    'P': '12',
    'Q': '13',
    'R': '14',
    'S': '15',
    'T': '16',
    'V': '17',
    'W': '18',
    'Y': '19',
    'X': '20',
}

_aa_1_3_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    '-': 'GAP',
    'X': 'URI',
}

_polar_dict = {
    'A': 0.,
    'C': 1.,
    'D': 1.,
    'E': 1.,
    'F': 0.,
    'G': 1.,
    'H': 1.,
    'I': 0.,
    'K': 1.,
    'L': 0.,
    'M': 0.,
    'N': 1.,
    'P': 0.,
    'Q': 1.,
    'R': 1.,
    'S': 1.,
    'T': 1.,
    'V': 0.,
    'W': 0.,
    'Y': 1.,
    'X': 0., 
}

_hydropathy_dict = {
    'A': 1.8,
    'C': 2.5,
    'D': -3.5,
    'E': -3.5,
    'F': 2.8,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'K': -3.9,
    'L': 3.8,
    'M': 1.9,
    'N': -3.5,
    'P': -1.6,
    'Q': -3.5,
    'R': -4.5,
    'S': -0.8,
    'T': -0.7,
    'V': 4.2,
    'W': -0.9,
    'Y': -1.3,
    'X': 0.0,
}

_vdw_volume_dict = {
    'A': 67.,
    'C': 86.,
    'D': 91.,
    'E': 109.,
    'F': 135.,
    'G': 48.,
    'H': 118.,
    'I': 124.,
    'K': 135.,
    'L': 124.,
    'M': 124.,
    'N': 96.,
    'P': 90.,
    'Q': 114.,
    'R': 148.,
    'S': 90.,
    'T': 93.,
    'V': 105.,
    'W': 163.,
    'Y': 141.,
    'X': 0.0,
}

_charge_dict = {
    'A': 0.,
    'C': 0.,
    'D': -1.,
    'E': -1.,
    'F': 0.,
    'G': 0.,
    'H': 1.,
    'I': 0.,
    'K': 1.,
    'L': 0.,
    'M': 0.,
    'N': 0.,
    'P': 0.,
    'Q': 0.,
    'R': 1.,
    'S': 0.,
    'T': 0.,
    'V': 0.,
    'W': 0.,
    'Y': 0.,
    'X': 0., 
}

def get_seq(pdb_file):
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    for chain in structure.get_chains():
        seq = seq1(''.join([residue.resname for residue in chain]))
    return seq

def get_chain_seq(pdb_file, chain_id=None):
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    chain_list = [_ for _ in chain_id]
    seq = ''
    for chain in structure.get_chains():
        if chain.get_id() in chain_list:
            for residue in chain:
                seq += residue.resname
    return seq1(seq)

def get_multi_seq(pdb_file):
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    residues = [r for r in structure.get_residues()]
    seq = seq1(''.join([residue.resname for residue in residues]))
    return seq

def get_complex_seq(pdb_file):
    seqs = dict()
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    for count, chain in enumerate(structure.get_chains()):
        if count == 0:
            id_ = 'A'
        elif count == 1:
            id_ = 'B'
        else:
            break
        seq = seq1(''.join([residue.resname for residue in chain]))

        seqs.update({id_: seq})
    return seqs

def get_pp_seq(pdb_file):
    p = PDBParser(QUIET=True)
    ppb=PPBuilder()
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    seq = []
    for pp in ppb.build_peptides(structure):
        seq.append(pp.get_sequence())
    return seq

def get_seq_prim(pdb_file):
    seq = get_seq(pdb_file)
    seq_prim = letter_to_num(seq, _aa_dict)
    return seq_prim

def num_to_letter(ints, dict_):
    dict_ = {v: k for k, v in dict_.items()}
    str_list = [dict_[str(_)] for _ in ints]
    string = ""
    for i in str_list:
        string += i
    return string

def letter_to_num(string, dict_):
    """Function taken from ProteinNet (https://github.com/aqlaboratory/proteinnet/blob/master/code/text_parser.py).
    Convert string of letters to list of ints"""
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num

def get_atom_coord(residue, atom_type):
    if atom_type in residue:
        return residue[atom_type].get_coord()
    else:
        return [0, 0, 0]

def get_cb_or_ca_coord(residue):
    if 'CB' in residue:
        return residue['CB'].get_coord()
    elif 'CA' in residue:
        return residue['CA'].get_coord()
    else:
        return [0, 0, 0]

def get_atom_coords_mask(coords):
    mask = torch.ByteTensor([1 if sum(_) != 0 else 0 for _ in coords])
    mask = mask & (1 - torch.any(torch.isnan(coords), dim=1).byte())
    return mask

def place_fourth_atom(a_coord: torch.Tensor, b_coord: torch.Tensor,
                      c_coord: torch.Tensor, length: torch.Tensor,
                      planar: torch.Tensor,
                      dihedral: torch.Tensor) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])
    return d_coord

def place_missing_cb_o(atom_coords):
    cb_coords = place_fourth_atom(atom_coords['C'], atom_coords['N'],
                                  atom_coords['CA'], torch.tensor(1.522),
                                  torch.tensor(1.927), torch.tensor(-2.143))
    o_coords = place_fourth_atom(
        torch.roll(atom_coords['N'], shifts=-1, dims=0), atom_coords['CA'],
        atom_coords['C'], torch.tensor(1.231), torch.tensor(2.108),
        torch.tensor(-3.142))

    bb_mask = get_atom_coords_mask(atom_coords['N']) & get_atom_coords_mask(
        atom_coords['CA']) & get_atom_coords_mask(atom_coords['C'])
    missing_cb = (get_atom_coords_mask(atom_coords['CB']) & bb_mask) == 0
    atom_coords['CB'][missing_cb] = cb_coords[missing_cb]

    bb_mask = get_atom_coords_mask(
        torch.roll(
            atom_coords['N'], shifts=-1, dims=0)) & get_atom_coords_mask(
                atom_coords['CA']) & get_atom_coords_mask(atom_coords['C'])
    missing_o = (get_atom_coords_mask(atom_coords['O']) & bb_mask) == 0
    atom_coords['O'][missing_o] = o_coords[missing_o]

def get_atom_coords(pdb_file):
    """read pdb file"""
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    residues = [r for r in structure.get_residues()]
    """get atom coords"""
    n_coords = np.array([get_atom_coord(r, 'N') for r in residues])
    ca_coords = np.array([get_atom_coord(r, 'CA') for r in residues])
    c_coords = np.array([get_atom_coord(r, 'C') for r in residues])
    cb_coords = np.array([get_atom_coord(r, 'CB') for r in residues])
    cb_ca_coords = np.array([get_cb_or_ca_coord(r) for r in residues])
    o_coords = np.array([get_atom_coord(r, 'O') for r in residues])
    """save to dict"""
    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    atom_coords['CB'] = cb_coords
    atom_coords['CBCA'] = cb_ca_coords
    atom_coords['O'] = o_coords
    """place missing CB and O atoms"""
    place_missing_cb_o(atom_coords)
    return atom_coords

def get_backbone_coords(pdb_file):
    """read pdb file"""
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    residues = [r for r in structure.get_residues()]
    """get atom coords"""
    n_coords = np.array([get_atom_coord(r, 'N') for r in residues])
    ca_coords = np.array([get_atom_coord(r, 'CA') for r in residues])
    c_coords = np.array([get_atom_coord(r, 'C') for r in residues])
    cb_ca_coords = np.array([get_cb_or_ca_coord(r) for r in residues])
    o_coords = np.array([get_atom_coord(r, 'O') for r in residues])
    """save to dict"""
    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    atom_coords['CB'] = cb_ca_coords
    return atom_coords

def get_chain_coords(pdb_file, chain_id=None):
    """read pdb file"""
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    chain_list = [_ for _ in chain_id]
    structure = p.get_structure(file_name, pdb_file)
    residues = []
    for chain in structure.get_chains():
        if chain.get_id() in chain_list:
            for residue in chain:
                residues.append(residue)
    """get atom coords"""
    n_coords = np.array([get_atom_coord(r, 'N') for r in residues])
    ca_coords = np.array([get_atom_coord(r, 'CA') for r in residues])
    c_coords = np.array([get_atom_coord(r, 'C') for r in residues])
    """save to dict"""
    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    return atom_coords

def get_coords(pdb_file):
    """read pdb file"""
    p = PDBParser(QUIET=True)
    file_name = splitext(basename(pdb_file))[0]
    structure = p.get_structure(file_name, pdb_file)
    residues = [r for r in structure.get_residues()]
    """get atom coords"""
    n_coords = [get_atom_coord(r, 'N') for r in residues]
    ca_coords = [get_atom_coord(r, 'CA') for r in residues]
    c_coords = [get_atom_coord(r, 'C') for r in residues]
    """save to dict"""
    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    return atom_coords

def load_full_seq(fasta_file):
    """Concatenates the sequences of all the chains in a fasta file"""
    with open(fasta_file, 'r') as f:
        return ''.join(
            [seq.rstrip() for seq in f.readlines() if seq[0] != '>'])

def get_fasta_chain_seq(fasta_file, chain_id):
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        if ":{}".format(chain_id) in chain.id:
            return str(chain.seq)

def get_rec_seq_len(fasta_file):
    r_len = len(get_fasta_chain_seq(fasta_file, "A"))

    return r_len

def save_PDB(out_pdb: str,
             coords: torch.Tensor,
             seq: str,
             delim: int = None) -> None:
    """
    Write set of N, CA, C, O, CB coords to PDB file
    """

    if type(delim) == type(None):
        delim = -1

    atoms = ['N', 'CA', 'C', 'O', 'CB']

    with open(out_pdb, "a") as f:
        k = 0
        for r, residue in enumerate(coords):
            AA = _aa_1_3_dict[seq[r]]
            for a, atom in enumerate(residue):
                if AA == "GLY" and atoms[a] == "CB": continue
                x, y, z = atom
                f.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f %4.2f\n"
                    % (k + 1, atoms[a], AA, "A" if r <= delim else "B", r + 1,
                       x, y, z, 1, 0))
                k += 1
        f.close()

def save_PDB_3(out_pdb: str,
               coords: torch.Tensor,
               seq: str,
               delim: int = None) -> None:
    """
    Write set of N, CA, C coords to PDB file
    """

    if type(delim) == type(None):
        delim = -1

    atoms = ['N', 'CA', 'C']

    with open(out_pdb, "w") as f:
        k = 0
        for r, residue in enumerate(coords):
            AA = _aa_1_3_dict[seq[r]]
            for a, atom in enumerate(residue):
                x, y, z = atom
                f.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f\n"
                    % (k + 1, atoms[a], AA, "A" if r <= delim else "B", r + 1,
                       x, y, z, 1))
                k += 1
        f.close()

def save_PDB_4(out_pdb: str,
               coords: torch.Tensor,
               seq: str,
               delim: int = None) -> None:
    """
    Write set of N, CA, C, CB coords to PDB file
    """

    if type(delim) == type(None):
        delim = -1

    atoms = ['N', 'CA', 'C', 'CB']

    with open(out_pdb, "w") as f:
        k = 0
        for r, residue in enumerate(coords):
            AA = _aa_1_3_dict[seq[r]]
            for a, atom in enumerate(residue):
                x, y, z = atom
                f.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f\n"
                    % (k + 1, atoms[a], AA, "A" if r <= delim else "B", r + 1,
                       x, y, z, 1))
                k += 1
        f.close()

def get_residue_depth(pdb_file):
    parser = PDBParser()
    file_name = splitext(basename(pdb_file))[0]
    structure = parser.get_structure(file_name, pdb_file)
    rd = []
    for chain in structure.get_chains():
        surface = get_surface(chain)
        for res in chain:
            rd.append(residue_depth(res, surface))
    return rd

def get_sasa(pdb_file):
    parser = PDBParser(QUIET=1)
    sr = ShrakeRupley()
    file_name = splitext(basename(pdb_file))[0]
    structure = parser.get_structure(file_name, pdb_file)
    chains = [c for c in structure.get_chains()]
    for chain in chains:
        sr.compute(chain, level="R")

    res1 = [r for r in chains[0]]
    res2 = [r for r in chains[1]]

    sasa1 = [r.sasa for r in res1]
    sasa2 = [r.sasa for r in res2]

    return sasa1, sasa2

def get_sasa_2(pdb_file):
    parser = PDBParser(QUIET=1)
    sr = ShrakeRupley()
    file_name = splitext(basename(pdb_file))[0]
    structure = parser.get_structure(file_name, pdb_file)
    sr.compute(structure, level="R")
    residues = [r for r in structure.get_residues()]
    sasa = [r.sasa for r in residues]

    return sasa
