import os
from collections import namedtuple

import numpy as np
import pandas as pd

import torch

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
morgan_generator = GetMorganGenerator(radius=1, fpSize=512)

def create_fingerprints(smiles):
    return np.stack([
        morgan_generator.GetFingerprintAsNumPy(MolFromSmiles(smile))
        for smile in smiles
    ])

def load_data(datadir, x_include=['embeddings', 'fingerprints'], tensor=False):

    data = {'train': {}, 'validation': {}, 'test': {}}

    # SMILES
    data['train']['smiles'] = pd.read_csv(os.path.join(datadir, 'train/smiles.csv')).to_numpy().squeeze()
    data['validation']['smiles'] = pd.read_csv(os.path.join(datadir, 'validation/smiles.csv')).to_numpy().squeeze()
    data['test']['smiles'] = pd.read_csv(os.path.join(datadir, 'test/smiles.csv')).to_numpy().squeeze()

    # Unimol Embeddings
    data['train']['embeddings'] = np.load(os.path.join(datadir, 'train/unimol_embeddings.npy'))
    data['validation']['embeddings'] = np.load(os.path.join(datadir, 'validation/unimol_embeddings.npy'))
    data['test']['embeddings'] = np.load(os.path.join(datadir, 'test/unimol_embeddings.npy'))

    # Morgan Fingerprints
    data['train']['fingerprints'] = create_fingerprints(data['train']['smiles'])
    data['validation']['fingerprints'] = create_fingerprints(data['validation']['smiles'])
    data['test']['fingerprints'] = create_fingerprints(data['test']['smiles'])

    # Immunomodulation Targets
    data['train']['y'] = pd.read_csv(os.path.join(datadir, 'train/y.csv')).to_numpy()
    data['validation']['y'] = pd.read_csv(os.path.join(datadir, 'validation/y.csv')).to_numpy()
    data['test']['y'] = pd.read_csv(os.path.join(datadir, 'test/y.csv')).to_numpy()

    # RDKit Targets
    try:
        data['train']['y_rdkit'] = pd.read_csv(os.path.join(datadir, 'train/y_rdkit.csv')).to_numpy()
        data['validation']['y_rdkit'] = pd.read_csv(os.path.join(datadir, 'validation/y_rdkit.csv')).to_numpy()
        data['test']['y_rdkit'] = pd.read_csv(os.path.join(datadir, 'test/y_rdkit.csv')).to_numpy()
    except: None

    # Immunomodulation Standard Errors
    try:
        data['train']['y_err'] = pd.read_csv(os.path.join(datadir, 'train/std.csv')).to_numpy()
        data['validation']['y_err'] = pd.read_csv(os.path.join(datadir, 'validation/std.csv')).to_numpy()
        data['test']['y_err'] = pd.read_csv(os.path.join(datadir, 'test/std.csv')).to_numpy()
    except: None

    return data

def to_tensor(data, device='cpu'):
    
    import torch
    
    for split in data.keys():
        for item in data[split].keys():
            try:
                data[split][item] = torch.tensor(
                    data[split][item], 
                    dtype=torch.float32
                ).detach().to(device)
            except:
                continue

    return data

def to_namedtuple(data):

    Splits = namedtuple('Splits', ['train', 'validation', 'test'])
    Data = namedtuple('Data', data['train'].keys())

    train = Data(**data['train'])
    validation = Data(**data['validation'])
    test = Data(**data['test'])

    return Splits(train=train, validation=validation, test=test)
