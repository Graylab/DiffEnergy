from argparse import ArgumentParser
import csv
from pathlib import Path
import warnings
from biotite.structure.io import load_structure
from natsort import natsort_key
from tqdm import tqdm

from diffenergy.dfmdock_tr.inference import get_sample_metrics
from diffenergy.dfmdock_tr.utils.metrics import METRICS_KEYS

#for if you need to generate a metrics file (dockq, rmsds, etc) after sampling
def write_sample_metrics(ground_truth,samples_folder,metrics_csv_out):
    ground_truth = Path(ground_truth)
    samples_folder = Path(samples_folder)
    metrics_csv_out = Path(metrics_csv_out)
    
    with open(metrics_csv_out,'w',newline='',buffering=1) as metrics_handle:
        writer = csv.DictWriter(metrics_handle,fieldnames=['index',*METRICS_KEYS])
        writer.writeheader()
        for gt in tqdm(list(ground_truth.glob("*.pdb"))):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gtstruct = load_structure(gt)
            name = gt.stem
            for sample in tqdm(sorted(samples_folder.glob(f"{name}_*.pdb"),key=natsort_key),leave=False,desc=name):
                metrics = get_sample_metrics(gtstruct,sample)
                writer.writerow({'index':sample.stem,**metrics})    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ground_truth")
    parser.add_argument("samples")
    parser.add_argument("metrics_csv")
    
    args = parser.parse_args()    
    write_sample_metrics(args.ground_truth,args.samples,args.metrics_csv)
    