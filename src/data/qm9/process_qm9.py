#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

import os


# ## Read XYZs

# In[2]:


labels = ['SMILES', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
bad_samples = set([
    21725, 87037, 59827, 117523, 128113, 
    129053, 129152, 129158, 130535, 6620, 
    59818, 21725, 59827, 128113, 129053, 
    129152, 130535, 6620, 59818
])


# In[3]:


csv_fname = 'qm9.csv'


# In[ ]:


with open(csv_fname, 'w') as f:
    f.write(','.join(labels) + '\n')

coordinates = []

for i, fname in tqdm(enumerate(os.listdir('xyz'))):
    fname = './xyz/' + fname

    with open(fname) as f: 
        lines = f.readlines()
    
    mol_id, *targets = lines[1].split()[1:]
    if mol_id in bad_samples: continue

    smiles = lines[-2].split()[-1:]

    # Generate Coordinates
    
    mol = Chem.MolFromSmiles(smiles[0])
    if mol is None: continue
    mol = Chem.AddHs(mol)
    embed = AllChem.EmbedMolecule(mol, randomSeed=16)  # Set a random seed for reproducibility
    if embed != 0: continue
    AllChem.UFFOptimizeMolecule(mol)  # Perform an optimization using UFF force field

    coords = np.full((9, 3), np.inf)

    j = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H': continue
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())  # Get coordinates for each atom
        coords[j] = pos.x, pos.y, pos.z
        j += 1

    coordinates.append(coords)
    
    with open(csv_fname, 'a') as f:
        f.write(','.join(smiles + targets) + '\n')


# In[ ]:


coordinates = np.stack(coordinates)


# ## Filter Data

# In[ ]:


data = pd.read_csv('qm9.csv')


# In[ ]:


# Filter Extreme Outliers
mask = (data['A'] < 500) & (data['A'] > 0)


# In[ ]:


data = data[mask]
coordinates = coordinates[mask]


# ## Transform

# In[ ]:


# Separate SMILES and targets
smiles = data['SMILES']
y = data.iloc[:, 1:]


# In[ ]:


# Log of A, B, C
y['A'] = np.log(y['A'])
y['B'] = np.log(y['B'])
y['C'] = np.log(y['C'])

y.rename({'A': 'logA', 'B': 'logB', 'C': 'logC'}, axis=1, inplace=True)


# In[ ]:


# Normalize
mu = y.mean()
std = y.std()
norm_y = (y - mu)/std


# In[ ]:


# Save mean and standard deviation
norm_statistics = pd.concat((mu, std), axis=1)
norm_statistics.columns = ['mean', 'std']
norm_statistics.to_csv('norm_statistics.csv')


# ## Shuffle

# In[ ]:


indices = np.random.permutation(len(smiles))


# In[ ]:


smiles_shuffled = smiles.iloc[indices]
norm_y_shuffled = norm_y.iloc[indices]
coordinates_shuffled = coordinates[indices]


# ## Save

# In[ ]:


smiles_shuffled.to_csv('smiles.csv', index=False)
norm_y_shuffled.to_csv('norm_y.csv', index=False)
np.save('coordinates.npy', coordinates_shuffled)


# In[ ]:


d = np.nan_to_num(
    np.linalg.norm(
        np.expand_dims(coordinates, axis=1) - np.expand_dims(coordinates, axis=2), 
        axis=-1
    ), 
    nan=np.inf, posinf=np.inf, neginf=np.inf
)


# In[ ]:


np.save('distances.npy', d)


# In[ ]:




