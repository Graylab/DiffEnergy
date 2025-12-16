from pathlib import Path
import pandas as pd
from natsort import natsort_key


def score_to_csv(score_file,csv_out):
    score_file = Path(score_file)
    csv_out = Path(csv_out)
    rosetta_df = pd.read_csv(score_file,sep=r'\s+',header=1) #whitespace separated file; first line is a dud
    del rosetta_df['SCORE:'] #each line begins with "SCORE:", so remove

    rosetta_df['id'] = rosetta_df['description'].str.rsplit("_0001",n=1).str[0]
    rosetta_df['pdb_id'] = rosetta_df['id'].str.split("_",n=1).str[0]
    rosetta_df = rosetta_df.sort_values('id',key=natsort_key)

    #move ids to the beginning cause I want them first in the csv
    cols = list(rosetta_df.columns)
    cols.remove('id'); cols.remove('pdb_id')
    cols = ['id','pdb_id',*cols]
    rosetta_df = rosetta_df[cols]

    rosetta_df.to_csv(csv_out,index=False)


if __name__ == "__main__":
    import sys
    score_to_csv(sys.argv[1],sys.argv[2])

