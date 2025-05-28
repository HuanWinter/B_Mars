# B_Mars - Mars' external magnetic field modeling

Scripts for a project with aim of modeling the  Mars' external magnetic field ($B_{1}$) using the MLP method.

If you have any questions w.r.t the code, please contact andong.hu@colorado.edu or Xiaohua.Fang@lasp.colorado.edu.

## Overview

This project is conducted primarily in python that present theory alongside application. 

Two folders 'Figs' (for saving figures) and 'Checkpoints' (for trained model) should be constructed automatically. 

To be able to run the python/notebooks themselves, it is most convenient to use the supplied conda environment file ([environment.yml](environment.yml)) to create the corresponding conda environment as described in Section 'Tutorial' or [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

### Python modules

The following files contain useful functions and classes used throughout the notebooks.

- [main function](Main.py) : main function, including preprocessing, modeling and plotting.
- [Functions](funs.py) : Various functions ranging from model training to plotting, including arritectures used for training.
- [Analysis](Analysis.ipynb) : Analysis the results and plot. 

The notebooks are used to display the outputs.

# Tutorial

## environment install 

    conda env create -f environment.yml
    
Then activate virtue environment by

    conda activate mars_ml_env

## end-to-end modeling

    python3 Main.py

## outputs analysis

running Analysis.ipynb


