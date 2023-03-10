## Exploratory work on intermediate ligand generation for FEP.

See notebook for list of FEP ligand pairs.

To use create a conda environment that contains the following packages: (is basically the same as the default dev biosimspace env, see https://biosimspace.org/install.html).

Dependencies:

- biosimspace
- rdkit
- matplotlib
- ipython/jupyter
- pandas
- obabel (conda install -c conda-forge openbabel)

# Usage
These scripts can be used to 
1. Generate intermediates:
This can be done using the command line with test.py (& display.ipynb for display)
The same can be done with network_enrichment_endpoints.ipynb (be aware that this notebook currently does not save the generated intermediates)

2. Write molecules to sdf & give identifier:
Using smiles_for_fep.py

3. Align intermediates:
Using align.ipynb