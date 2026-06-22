import datetime
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Mapping
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from diffenergy.inference import handle_overwrite_dir, strip_keys, write_config
from rosetta_refine.score_to_csv import score_to_csv
import shutil



## Python version of the following rosetta bash script:


# ml gcc #I don't think this is necessary? since rosetta is already compiled

# # job description
# ROSETTABIN=/scratch16/jgray21/aharmal1/Rosetta/main/source/bin
# ROSETTAEXE=docking_protocol
# COMPILER=mpi.linuxgccrelease
# EXE=$ROSETTABIN/$ROSETTAEXE.$COMPILER
# echo Starting MPI job running $EXE

# mkdir -p ../rosetta_score/dfmdock_trtrained_deterministic/refined_pdbs

# # only one structure per sample since it already takes forever
# date
# time mpirun $EXE \
#     -in:file:l dfmdock_sample_index.txt \
#     -out:file:scorefile ../rosetta_score/dfmdock_trtrained_deterministic/refined_score.sc \
#     -out:path:pdb ../rosetta_score/dfmdock_trtrained_deterministic/refined_pdbs \
#     -nstruct 1 \
#     @score_refine_flags
# date

# python score_to_csv.py ../rosetta_score/dfmdock_trtrained_deterministic/refined_score.sc ../rosetta_score/dfmdock_trtrained_deterministic/dfmdock/refined_score.csv

ROSETTABIN = "/PATH/TO/ROSETTA/BIN"
ROSETTAEXE = "docking_protocol"
COMPILER = "mpi.linuxgccrelease"

@hydra.main(version_base=None, config_path="../configs")
def main(config:DictConfig):
    rosetta_bin = config.get("ROSETTABIN",ROSETTABIN)
    rosetta_exe = config.get("ROSETTAEXE",ROSETTAEXE)
    rosetta_compiler = config.get("COMPILER",COMPILER)

    postprocess_only = config.get("postprocess_only",False)

    out_dir = Path(config.get("out_dir"))
    if not postprocess_only:
        if out_dir.exists() and not config.get("resume_existing"):
            handle_overwrite_dir(out_dir,config.get("overwrite_output",False))
        
        out_dir.mkdir(exist_ok=True,parents=True)

        print(OmegaConf.to_yaml(config))
        write_config(config,out_dir/'config.yaml',require_compatible_if_existing=True,extra_ignore_keys=['ROSETTABIN'])
    else:
        assert out_dir.exists()
        assert (out_dir/'config.yaml').exists()


    pdb_list = Path(config.get("pdb_list"))
    
    ## optional parent folder for paths specified in pdb_list. If specified, program will prepend pdb_dir to filename
    pdb_dir = config.get("pdb_dir",None) 
    pdb_dir = Path(pdb_dir) if pdb_dir is not None else None

    ## Problem: Rosetta only records the filenames of the pdbs it analyzes, not the full path nor a user-provided id. To make sure
    ## the output is deduplicated, we're gonna do something a little silly: create a folder of symlinks with sequential filenames
    ## and use them to index into an array of ids after processing is done.
    
    ## First, read the input file
    if pdb_list.suffix == 'txt':
        pdb_df = pd.read_csv(pdb_list,header=None).rename(columns={0:"filename"}) #read "csv" file with no header
    else:
        pdb_df = pd.read_csv(pdb_list)


    ## Then, try to infer the proper pdb / sample id. If the input is a csv with an id or index column, use that; otherwise, use the filename 
    ## (w/o prepending pdb_dir)
    if 'id' in pdb_df.columns: ids = pdb_df['id']
    elif 'index' in pdb_df.columns: ids = pdb_df['index']
    else: ids = pdb_df['filename'] 
    
    ## prepend pdb_dir to filename
    pdbs = pdb_df['filename']
    if pdb_dir is not None:
        pdbs = (str(pdb_dir.absolute()) + "/") + pdbs

    out_score_file = out_dir/'refined_score.sc'
    out_score_csv = out_score_file.with_suffix('.csv')
    out_pdb_dir = out_dir/'refined_pdbs'
    out_pdb_dir.mkdir(exist_ok=True,parents=True)

    if not postprocess_only: #Allow for the processing of a score file w/o re-running rosetta

        ## Rosetta automatically skips files it's already created! so no need for a skiplist. However, we do need to remove any .in_progress files,
        ## However, if we ended a previous run early, we might have leftover "*.in_progress" files. By default, Rosetta will skip these,
        ## thinking they belong to another process. By default, we remove these in_progress files, assuming we are the only process.
        ## setting config.remove_in_progress to False allows running this file multiple times on the same output, though I can't guarantee
        ## post-processing will work perfectly so you might want to run the script again with postprocess_only=True to make sure the score csv
        ## file is properly created
        if out_pdb_dir.exists() and config.get("remove_in_process",True):
            for f in out_pdb_dir.glob("*.in_progress"):
                f.unlink()

        with TemporaryDirectory() as temp_dir: #temporary directory for fake index list and symlink folder
            symlink_folder = Path(temp_dir)/"pdbs"
            symlink_folder.mkdir()
            rosetta_pdb_list = Path(temp_dir)/"pdb_list.txt"

            print("temporary pdb list @",rosetta_pdb_list)

            ## create rosetta input file with unique-named pdb paths
            with open(rosetta_pdb_list,"w") as f:
                for index, filename in pdbs.items():
                    p = symlink_folder/f"{index}.pdb"
                    os.symlink(filename,p)
                    f.write(f"{p}\n")
            
            flags_file = config.get("flags_file",None)
            flags_dict = config.get("flags",{})
            assert isinstance(flags_dict,Mapping) #dict of rosetta flags

            flags_list = []
            for k,v in flags_dict.items():
                flags_list.append(f"-{k}")
                flags_list.append(str(v))
            if flags_file is not None:
                flags_list.append(f"@{flags_file}")

            rosetta_exe = f"{rosetta_bin}/{rosetta_exe}.{rosetta_compiler}"
            command = ['time', 'mpirun', rosetta_exe,
                    '-in:file:l',rosetta_pdb_list,
                    '-out:file:scorefile',out_score_file,
                    '-out:path:pdb',out_pdb_dir] + flags_list

            print(f"Beginning rosetta docking at {datetime.datetime.now()}",flush=True)

            process = subprocess.run(command)
            if process.returncode == 255:
                ## Most rosetta errors are things like fastrelax failed - usually fine, we'll just miss a structure or two
                print("Rosetta errors were encountered! Continuing with postprocessing.")
            else:
                process.check_returncode()

            print(f"Rosetta docking complete at {datetime.datetime.now()}, processing score file into csv...")
    else:
        print("postprocess_only flag set in config; skipping rosetta")

    rosetta_df = score_to_csv(out_score_file,None)
    rosetta_df['id'] = rosetta_df['id'].astype(int)
    rosetta_df.set_index('id',drop=True)

    rosetta_df['id'] = ids.loc[rosetta_df.index] #set real ids using the index. Note that there might be gaps in rosetta_df from rosetta errors, so use rosetta_df to index into ids instead of vice-versa
    rosetta_df['rosetta_id'] = rosetta_df['id'].str.cat(rosetta_df['rosetta_index'],sep='_') #reconstruct proper rosetta id
    rosetta_df['pdb_id'] = rosetta_df['id'].str.split("_",n=1).str[0] #reconstruct proper pdb id
    rosetta_df = rosetta_df.sort_values(['id','rosetta_index'])
    
    lead_cols = ['rosetta_id','id','pdb_id','rosetta_index','filename']
    rosetta_df = rosetta_df[lead_cols + [d for d in rosetta_df.columns if d not in lead_cols]]
    rosetta_df.to_csv(out_score_csv,index=False)

    print(f"Score postprocessing complete.")

if __name__ == "__main__":
    main()