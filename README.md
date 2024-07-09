# Viscous Gubser BSQ

This repository provides the semi-analytic solution for Gubser Flow with BSQ conserved charges.

There are two branches:
  - `qgp_eos`
  - `ccake_eos`
Each implements a different equation of state (EoS) corresponding the the EoS's discussed in the [paper]().
The scripts that generate the images in the paper are:
  - `qgp_eos`:`plots/compare_mus.py` -- this generates Figures 1 and 2 in the paper
    - To run type `python compare_mus.py` in the appropriate directory  
  - `qgp_eos`:`plots/freezeout-surface.py` -- this generates Figure 3
    - To run type `freezeout-surface.py` in the appropriate directory  
  - `ccake_eos`:`plots/freezeout-surface.py` -- this generates Figure 6
  - `ccake_eos`:`plots/for_ccake_plot/plot_gubser_figure_4.py` -- this generates Figure 5 (sorry for confusing naming)
    - To run type `python plot_gubser_figure_4.py <analytic_dir> <sim_dir>` in the appropriate directory  

The scripts that are used to generate initial conditions for and analytic evolutions is:
  - `ccake_eos`:`plots/for_ccake/generate_initial_conditions.py` -- generates the initial conditions for CCAKE to run gubser test
  - `ccake_eos`:`plots/for_ccake/generate_semi-analytic_solution.py` -- generates the the semi-analytic solution to compare agains numerical ones

These files work with the `run.cfg` file, which specifies the equation of state parameters, initial conditions, and output directory.
To run the generation scripts, running `python generate_<>.py` will be enough.
    
