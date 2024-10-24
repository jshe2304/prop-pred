import numpy as np

#import pubchempy
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AllChem

class Fingerprinter:
    def __init__(self):
    
        self.morgan = AllChem.GetMorganGenerator(radius=2)

    def __call__(self, smiles):
        '''
        Fingerprint a list of SMILES
        '''
        if type(smiles) is not list:
            raise Exception('Argument should be list of SMILES strings.')

        fingerprints = []

        for smile in smiles:

            morgan = self.morgan.GetFingerprintAsNumPy(MolFromSmiles(smile))
            #pubchem = np.fromiter(pubchempy.get_compounds(smile, 'smiles')[0].cactvs_fingerprint, dtype='uint8')

            fingerprints.append(morgan)

        return np.stack(fingerprints)
