import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from unimol_tools import MolPredict, UniMolRepr

data_path = '../../data/immunomodulation/log_normalized/regression/'
model_path = './models/v1'

############################################################
# DATA 
############################################################

train_X = pd.read_csv(data_path + 'train/smiles.csv').to_numpy().squeeze().tolist()
validation_X = pd.read_csv(data_path + 'validation/smiles.csv').to_numpy().squeeze().tolist()
test_X = pd.read_csv(data_path + 'test/smiles.csv').to_numpy().squeeze().tolist()

############################################################
# MODEL 
############################################################

unimol = MolPredict(load_model=model_path)
unimol.predict(data=data_path + 'test.csv')

############################################################
# GET EMBEDDING 
############################################################

embedder = UniMolRepr(data_type='molecule', remove_hs=False)
embedder.model = unimol.model.model

train_embeddings = np.array(embedder.get_repr(train_X, return_atomic_reprs=True)['cls_repr'])
validation_embeddings = np.array(embedder.get_repr(validation_X, return_atomic_reprs=True)['cls_repr'])
test_embeddings = np.array(embedder.get_repr(test_X, return_atomic_reprs=True)['cls_repr'])

np.save(data_path + 'train/unimol_embeddings.npy', train_embeddings)
np.save(data_path + 'validation/unimol_embeddings.npy', validation_embeddings)
np.save(data_path + 'test/unimol_embeddings.npy', test_embeddings)
