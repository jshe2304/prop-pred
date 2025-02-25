import torch

from rdkit import Chem
from rdkit.Chem import MolFromSmiles

def smiles_to_graphs(smiles):

    for smile in smiles:

        None