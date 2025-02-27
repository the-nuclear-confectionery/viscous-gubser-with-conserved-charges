# Plotting scripts

## Usage
The following scripts were used to produce the figures in [PRD **\<VOLUME NO.\>**, \<IDENTIFIER\> (2025)](), [arXiv:2502:XXXX [nucl-th]](). The scripts all together using the [plot_vgcc.py](), i.e., running the command

```terminal
python plot_vgcc.py
```

or can be executed one by one as described below.

## `plot_Fig1_VGCC_T_mu_pi.py`
```terminal
python plot_Fig1_VGCC_T_mu_pi.py
```

## `plot_Fig2_VGCC_e_n_s.py`
```terminal
python plot_Fig2_VGCC_e_n_s.py
```

## `plot_Fig3_VGCC_hypersurface_and_trajectories.py`
```terminal
python plot_Fig3_VGCC_hypersurface_and_trajectories.py
```

## `plot_Fig4_CCAKE_vs_VGCC_e_n_u_pi_Rn.py`
```terminal
python plot_Fig4_CCAKE_vs_VGCC_e_n_u_pi_Rn.py --analytical_path <path/to/analytical/solution/directory> --simulation_path <path/to/numerical/simulation/results/directory> --output_path <path/to/output/directory>
```

## `plot_Fig5_CCAKE_vs_VGCC_hypersurface.py`
```terminal
python plot_Fig5_CCAKE_vs_VGCC_hypersurface.py --simulation_path <path/to/numerical/simulation/results/directory> --output_path <path/to/output/directory>
```