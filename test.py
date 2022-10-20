import pandas as pd
from pip import main

from rdkit import Chem, DataStructs
from rdkit.Chem import rdmolops, rdMolAlign
from rdkit.Chem import Draw, rdFMCS, AllChem, rdmolfiles, Descriptors, rdchem, rdMolDescriptors, rdmolops, rdFMCS
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Draw
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from rdkit.Chem.Draw import IPythonConsole

from IPython.display import display
from IPython.display import SVG,Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from SMintermgen import GenInterm
import time
from random import sample
from lomapscore import quantify_change
import pickle

# set perts to investigate in this list. make sure tgt/ligand names are correct because these strings are used to 
# parse ./ligands/.


def generateIntermediate(liga, ligb):
    """Function for generating a ligand intermediate between two ligand endpoints. Currently 
    only does SMILES-based intermediate generation"""
    # create instance of SMILES-based intermediate generator class, 
    # If insert_small = false, only remove tokens from the larger of the pair. If true, also insert tokens from smaller fragment
    # Takes ~10 minutes for whole set
    generator = GenInterm(insert_small = True)
    
    # returns df with the generated intermediates
    pair = [liga, ligb]
    
    df_interm = generator.generate_intermediates(pair)
    intermediates = df_interm['Intermediate'].tolist()

    return intermediates

def scoreIntermediates(liga, ligb, intermediates): 
    smiles_dict = {}

    if len(intermediates) > 0:
        #if len(intermediates) > 20:
            #sample intermediates, currently done at random
            #intermediates = sample(intermediates, 20)
        for intermediate in intermediates:
            try:
                change_score = quantify_change(liga, 
                                intermediate, 
                                ligb,
                                single_top= False)
                smiles_dict[intermediate] = change_score
            except ValueError:
                print("Skipping potential median; invalid structure.")
    
    if len(smiles_dict.keys()) > 0:
        top_median = max(smiles_dict, key = smiles_dict.get)
        return top_median, max(smiles_dict.values())
    else:
        return Chem.MolFromSmiles("c1ccccc1"), None

if __name__== "__main__":
    perts_to_intrap = perts_to_intrap = [
    ["tyk2", "ejm_54~ejm_31"], # start with a few easy FEPs.
    ["tyk2", "ejm_42~ejm_44"],
    ["tyk2", "ejm_44~ejm_31"],
    
    ["tyk2", "jmc_27~ejm_54"], # now try a few ring jumps.
    ["tyk2", "ejm_43~ejm_47"],
    ["tyk2", "ejm_49~ejm_45"],
    ["tnks2", "5p~5m"],
    ["tnks2", "1b~3a"],
    
    ["tnks2", "1a~8a"], # charge jumps probably out of scope.
    
    ["eg5", "CHEMBL1084678~CHEMBL1085666"], # some challenging multi-R-group ones.
    ["eg5", "CHEMBL1077227~CHEMBL1096002"], 
    ["eg5", "CHEMBL1082249~CHEMBL1085666"],
    
    ["galectin", "07_ligOH~05_ligOEt"], # tests a large MCS
    
    ["cats", "CatS_165~CatS_132"],  # also large MCS, but also large transform.
    ["cats", "CatS_29~CatS_141"], # mostly interested in seeing what happens to sulfoxide
    ["jnk1", "18635-1~18636-1"], # another MCS check. This transformation is already as 'small as can be'
    ["jnk1", "18627-1~18625-1"], # what if the transformation is already small as can be; but also just para->ortho?
    ["jnk1", "18634-1~18659-1"], # together with next; do both directions (A->B/B->A) result in same intermediate?
    ["jnk1", "18659-1~18634-1"]
                ]
    all_intermediatess = []
    top_intermediates = []
    lomap_scores = []
    for tgt, pert in perts_to_intrap:
        first_stamp = int(round(time.time() * 1000))
        # get the endpoint molecular objects.
        liga, ligb = pert.split("~")
        try:
            liga, ligb = [ Chem.rdmolfiles.SDMolSupplier(f"ligands/{tgt}/{lig}.sdf")[0] for lig in [liga, ligb]]
        except OSError:
            # naming is inconsistent; try with 'lig_' prefix.
            liga, ligb = [ Chem.rdmolfiles.SDMolSupplier(f"ligands/{tgt}/lig_{lig}.sdf")[0] for lig in [liga, ligb]]   
        # generate the intermediate.
        all_intermediates = generateIntermediate(liga, ligb) 
        all_intermediatess.append(all_intermediates)
        top_intermediate, lomap_score = scoreIntermediates(liga, ligb, all_intermediates)

        top_intermediates.append(Chem.MolToSmiles(top_intermediate))
        lomap_scores.append(lomap_score)

        with open("all_intermediates", "wb") as fp:   #Pickling
            pickle.dump(all_intermediatess, fp)

        with open("top_intermediates", "wb") as fp:   #Pickling
            pickle.dump(top_intermediates, fp)

        with open("lomap_scores", "wb") as fp:   #Pickling
            pickle.dump(lomap_scores, fp)

    print(top_intermediates)

    

    