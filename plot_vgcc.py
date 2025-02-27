#!/usr/bin/env python3

import os
import sys
import subprocess

def main():
    # Define the list of plotting scripts and their arguments.
    # Adjust the script names and arguments as needed.
    scripts_to_run = [
        # {
        #     "script": os.path.join("plots", "plot_Fig1_VGCC_T_mu_pi.py"),
        #     "args": ["--save", "plots/fig1_VGCC_T_mu_pi.png"]
        # }#,
        # {
        #     "script": os.path.join("plots", "plot_Fig3_VGCC_freezeout_and_trajectories.py"),
        #     "args": ["--freezeout", "true", "--traj", "output"]
        # },
        {
            "script": os.path.join("plots", "plot_Fig4_CCAKE_vs_VGCC_e_n_u_pi_Rn.py"),
            "args": ["--analytical_path", "data/output/analytic_solutions/", "--simulation_path", "data/CCAKE_results/", "--output_path", "data/output/plots/"]
        },
        {
            "script": os.path.join("plots", "plot_Fig5_CCAKE_vs_VGCC_hypersurface.py"),
            "args": ["--simulation_path", "data/CCAKE_results/", "--output_path", "data/output/plots/"]
        }
    ]
    
    for entry in scripts_to_run:
        script_path = entry["script"]
        args = entry["args"]
        
        if not os.path.exists(script_path):
            print(f"Script not found: {script_path}")
            continue
        
        cmd = ["python3", script_path] + args
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Finished running {script_path}\n")
        except subprocess.CalledProcessError as err:
            print(f"Error running {script_path}: {err}")
            sys.exit(1)

if __name__ == "__main__":
    main()

