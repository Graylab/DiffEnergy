from pathlib import Path
import pandas as pd
from natsort import natsort_key


def score_to_csv(score_file,csv_out=None)->pd.DataFrame:
    score_file = Path(score_file)
    rosetta_df = pd.read_csv(score_file,sep=r'\s+',header=1) #whitespace separated file; first line is a dud
    del rosetta_df['SCORE:'] #each line begins with "SCORE:", so remove

    rosetta_df.rename(columns={'description':'rosetta_id'},inplace=True)
    
    rosetta_df['rosetta_index'] = rosetta_df['rosetta_id'].str.rsplit("_",n=1).str[1]
    rosetta_df['id'] = rosetta_df['rosetta_id'].str.rsplit("_",n=1).str[0]
    rosetta_df['pdb_id'] = rosetta_df['id'].str.split("_",n=1).str[0]
    rosetta_df = rosetta_df.sort_values('id',key=natsort_key)
    rosetta_df['filename'] = rosetta_df['rosetta_id'] + ".pdb"

    #move ids to the beginning cause I want them first in the csv
    cols = list(rosetta_df.columns)
    cols.remove('id'); cols.remove('pdb_id')
    cols = ['id','pdb_id',*cols]
    rosetta_df = rosetta_df[cols]

    if csv_out is not None:
        csv_out = Path(csv_out)
        rosetta_df.to_csv(csv_out,index=False)
    
    return rosetta_df


if __name__ == "__main__":
    import sys
    score_to_csv(sys.argv[1],sys.argv[2])

