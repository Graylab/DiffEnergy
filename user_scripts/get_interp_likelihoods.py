from io import TextIOWrapper
import subprocess

from tqdm import tqdm

cfile = "likelihood_gaussian_1d_gtscore_diff_interp.yaml"

nsteps = 5000
samples_stem = f"../sample_results/trinormal_1d_smax70_{nsteps}"
# data_samples: ../sample_results/trinormal_1d_smax70.csv
# trajectory_index_file: ../sample_results/trinormal_1d_smax70_traj/trajectory_index_1000.csv

command = lambda i: ["python", 
                     "scripts/likelihood_gaussian_1d.py", 
                     f"--config-name={cfile}", 
                     f"++out_dir='../likelihood_results/likelihoodv3/interp_tests/3integrand_diff_{i}interp_1000_{nsteps}steps'", 
                     f"++num_interpolants={i}","++overwrite_output=True",
                     f"++data_samples={samples_stem + '.csv'}", f"++trajectory_index_file={samples_stem + '_traj/trajectory_index_1000.csv'}"]

files:list[TextIOWrapper] = []
processes:list[subprocess.Popen] = []
try:
    for i in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,75,100]:
        file = open(f"interp_logs/log_{i}.txt","w")
        files.append(file)
        processes.append(subprocess.Popen(command(i),stdout=file))
    for p in tqdm(processes):
        p.wait()
finally:
    for p in processes: p.kill()
    for f in files: f.close()