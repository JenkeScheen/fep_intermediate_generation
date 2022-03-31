## currently working with input = sdf file with molecules for which you want to calculate all the intermediates
## output is sdf file with intermediates for each pair?

## functionalities to add
## add function to handle case where R group consists of multiple fragments
## weld_r_groups gives index & aromaticity errors & sometimes returns SMILES with fragments & also for different rdkit versions get different number of intermediates back
## probably better to work with something else than dataframe for intermediates from each better

## functions that are in this class that might be useful to use from src/shared instead:
## find_largest()
## get_rgroups()
## also reading of sdf file might be done elsewhere

from cmath import nan
from rdkit import rdBase, Chem
rdBase.DisableLog('rdApp.*')
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRGroupDecomposition as rdRGD
from rdkit.Chem import PandasTools
from openbabel import pybel


import pandas as pd
import itertools
import re
import os

import itertools

from collections import defaultdict
from rdkit.Chem.rdchem import EditableMol


class GenInterm():
    def __init__(self, insert_small):
        self.insert_small = insert_small


    def generate_intermediates(self, pair):
        self.pair = pair
        self.find_largest()
        self.get_rgroups()
        # for each r-group
        for self.column in self.columns:
            self.tokenize()
            self.charge_original = self.return_charge(self.rgroup_large)
            self.permutate()
        self.weld()
        self.remove_fragmented()
        
        return self.df_interm
        
    def find_largest(self):
        """ 
        Identify the largest molecule of pair
        """
        ## currently selected based on num atoms
        larger = self.pair[0].GetNumAtoms() > self.pair[1].GetNumAtoms()
        d = {'Molecule': self.pair, 'Largest': [larger, not larger]}
        self.df_largest = pd.DataFrame(data=d)

    
    def get_rgroups(self):
        """ 
        Use maximum common substructure of two molecules to get the differing R-Groups
        """
        self.multiple = False
        # find maximim common substructure & pass if none found
        ## possible to use different comparison functions
        res=rdFMCS.FindMCS(self.pair, matchValences=True, ringMatchesRingOnly=True)

        core = Chem.MolFromSmarts(res.smartsString)
        res,_ = rdRGD.RGroupDecompose([core],self.pair,asSmiles=True,asRows=False)
        self.df_rgroup= pd.DataFrame(res)

        if len(self.df_rgroup.columns) > 2:
            self.multiple = True
            self.columns = self.df_rgroup.columns[1:]
        else:
            self.columns = ['R1']

        # remove hydrogens post match
        for column in self.columns:
            self.df_rgroup.loc[0,column] = self.remove_Hfragmens(self.df_rgroup.loc[0,column])
            self.df_rgroup.loc[1,column] = self.remove_Hfragmens(self.df_rgroup.loc[1,column])
        # combine largest and rgroup info
        self.df = pd.concat([self.df_largest, self.df_rgroup], axis=1)   


    def remove_Hfragmens(self, SMILES):
        tokens = SMILES.split('.')
        SMILES_stripped = [item for item in tokens if not re.match(r"(\[H\]\[\*\:.\])", item)]

        return '.'.join(SMILES_stripped)


    def tokenize(self):
        """ 
        Tokenize SMILES of the small and large R group
        """
        # set regex for SMILES 
        pattern =  r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        # get index of the largest molecule
        idx_large = self.df.index[self.df['Largest'] == True]
        idx_small = self.df.index[self.df['Largest'] == False]

        self.rgroup_large = self.df.at[idx_large[0],self.column]
        rgroup_small = self.df.at[idx_small[0],self.column]

        # for each r group delete one token and save as intermediate
        self.tokens = [token for token in regex.findall(self.rgroup_large)]
        self.tokens_small = [token for token in regex.findall(rgroup_small)]
        

    def permutate(self):
        """ 
        Remove tokens or edit tokens from R-groups of the largest molecule. Currently only able to remove tokens
        """
        # create new dataframe for storing adapted rgroups [has Core, R1, ... Rn]
        self.df_interm = pd.DataFrame(columns = self.df.columns[1:]).drop(columns = ['Largest'])

        # sample some/all options where tokens from small r-group are inserted
        available_small = [item for item in self.tokens_small if not re.match(r"(\[\*\:.\]|\.)", item)]

        if len(available_small) == 0:
            insert_small = False 
        else:
            to_add = set()
            for i in range(1, len(available_small)+1):
                for subset in itertools.combinations(available_small, i):
                    to_add.add(subset)
            insert_small = self.insert_small

        # get all the possible options for shorter r-group   
        # for large rgroup go over all the combinations with length in range 1 shorter than largest fragment - to 1 larger than shortest fragment 
        ## ask willem if shortest intermediate fragment can have as much atoms as shortest fragment or should always be 1 bigger
        ## maybe handle connection token differently
        for i in range(len(self.tokens) - 1, len(self.tokens_small) - 1, -1):
            for subset in itertools.combinations(self.tokens, i):
                # in some cases connection token will be removed, discard those cases
                ## does not take into account situation with multiple connections in rgroup, like for pair 7 
                ## C1CC1[*:1].[H][*:1].[H][*:1]
                connection = [item for item in subset if re.match('\\[\\*:.\\]', item)]
                if connection:
                    # add fragments of small subset into large subset
                    subsets = []
                    subsets.append(subset)

                    if insert_small:
                        for to_insert in to_add:
                            # only insert tokens from small fragment when smaller than long fragment
                            if len(subset) >= len(self.tokens) - len(to_insert): continue
                            for j in range(len(subset)):
                                a = list(subset)
                                a[j:j] = to_insert
                                subsets.append(a)

                    for subset in subsets:
                        interm = ''.join(subset)
                        # keep fragments with valid SMILES
                        if Chem.MolFromSmiles(interm) is not None:
                            # keep fragments that do not introduce/loose charge
                            # using openbabel & looking at disconnected rgroups could sometimes be incorrect
                            if self.check_charge(interm):
                                self.df_interm.loc[self.df_interm.shape[0], self.column] = interm
        
        # drop duplicate R groups to save time
        self.df_interm = self.df_interm.drop_duplicates(subset=self.column)   

        self.df_interm['Core'] = self.df.at[0,'Core']
        # in case of multiple rgroups also add unchanged rgroups to df
        if self.multiple == True:
            for rgroup in self.columns:
                if rgroup != self.column:
                    self.df_interm[rgroup] = self.df.at[0,rgroup]


    def check_charge(self, interm):
        charge = self.return_charge(interm)

        return charge == self.charge_original
    

    def return_charge(self, rgroup):
        try:
            mol = pybel.readstring("smi", rgroup)
            #bool polaronly, bool correctForPH, double pH
            mol.OBMol.AddHydrogens(False, True, 7.4)
            rgroup = mol.write("smi")

            mol = Chem.MolFromSmiles(rgroup)
            charge = Chem.GetFormalCharge(mol)
            # - for neutral, positive numbers for positive charge, negative for neragive charge
        except:
            # in case where pybel cannot read molecule, decided to kick out molecule
            charge = None
        return charge
    

    def weld(self):
        """ 
        Put modified rgroups back on the core, returns intermediate
        """
        self.df_interm['Intermediate'] = nan
        
        for index, row in self.df_interm.iterrows():
            try:
                combined_smiles = row['Core']
                for column in self.columns:
                    combined_smiles = combined_smiles + '.' + row[column]
                mol_to_weld = Chem.MolFromSmiles(combined_smiles)
                welded_mol = self.weld_r_groups(mol_to_weld)
                self.df_interm.at[index, 'Intermediate'] = welded_mol
            except AttributeError:
                pass
            except IndexError:
                pass
            except Chem.rdchem.AtomKekulizeException:
                pass

        ## not 100% sure this works on mol objects
        self.df_interm = self.df_interm.drop_duplicates(subset='Intermediate')
        self.df_interm = self.df_interm.drop(columns = [* ['Core'], *self.columns])
        self.df_interm = self.df_interm.dropna()
        

    def weld_r_groups(self, input_mol):
        """Adapted from 
        https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/CANPfuvAqWxR%2BosdH6TUT-%2B1Fy85fUXh0poRddrEQDxXmguJJ7Q%40mail.gmail.com/"""
        # First pass loop over atoms and find the atoms with an AtomMapNum
        join_dict = defaultdict(list)
        for atom in input_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                join_dict[map_num].append(atom)

        # Second pass, transfer the atom maps to the neighbor atoms
        for idx, atom_list in join_dict.items():
            if len(atom_list) == 2:
                atm_1, atm_2 = atom_list
                nbr_1 = [x.GetOtherAtom(atm_1) for x in atm_1.GetBonds()][0]
                nbr_1.SetAtomMapNum(idx)
                nbr_2 = [x.GetOtherAtom(atm_2) for x in atm_2.GetBonds()][0]
                nbr_2.SetAtomMapNum(idx)

        # Nuke all of the dummy atoms
        new_mol = Chem.DeleteSubstructs(input_mol, Chem.MolFromSmarts('[#0]'))

        # Third pass - arrange the atoms with AtomMapNum, these will be connected
        bond_join_dict = defaultdict(list)
        for atom in new_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                bond_join_dict[map_num].append(atom.GetIdx())

        # Make an editable molecule and add bonds between atoms with correspoing AtomMapNum
        em = EditableMol(new_mol)
        for idx, atom_list in bond_join_dict.items():
            if len(atom_list) == 2:
                start_atm, end_atm = atom_list
                em.AddBond(start_atm, end_atm,
                            order=Chem.rdchem.BondType.SINGLE)

        final_mol = em.GetMol()

        # remove the AtomMapNum values
        for atom in final_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        final_mol = Chem.RemoveHs(final_mol)

        return final_mol


    def remove_fragmented(self):
        """ 
        Removes intermediates that contain multiple fragments
        """
        self.df_interm['Fragmented'] = self.df_interm.apply(lambda row: '.' in Chem.MolToSmiles(row.Intermediate), axis=1)
        self.df_interm = self.df_interm[self.df_interm['Fragmented'] == False]

