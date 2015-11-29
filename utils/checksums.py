import os
import json
import pickle
import hashlib

import numpy as np


def hash_update(f, h, force_update=False):
    """Add a new filename and hash to `hashes.json`

    Parameters
    ----------
    f : str
        The filename (without a path) for the particular object
    h : str
        The hash for `f`
    force_update : bool
        Whether to replace 

    Returns
    -------
    None

    """
    hashes = 'data/hashes.json'
    kv = {f : h}
    if os.path.isfile(hashes):
        with open(hashes) as f:
            data = json.load(f)

        if f in data.keys():
            if force_update:
                data[f] = h
        else:
            data.update(kv)
        with open(hashes, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(hashes, 'w') as f:
            json.dump(kv, f, indent=4)
        

def hash_get(f):
    """Get the MD5 hash based on the rounded sum of the matrix values.

    Parameters
    ---------
    f : str
        The (relative path) filename for the particular object

    Returns
    -------
    m.hexdigest: str
        The digest containing only hexadecimal digits
    """
    X = pickle.load(open(f, 'rb'))
    if f == 'data/datamatrix.pkl':
        X = round(X.sum(), 8)
    elif f == 'data/pca_dict_50.pkl':
        X = np.round(X['X_reduced'][0].sum(), 8)
    m = hashlib.md5()
    m.update(X)
    return m.hexdigest()
    
def hash_same(k):
    """Get the hash and check it against what's in checksums.json.

    Parameters
    ----------
    k : str
        The filename (without a path) for the particular object

    Returns
    -------
    valid : bool
        True if the hashes match, False otherwise
    """
    j = json.load(open('data/hashes.json', 'r'))
    if k in j.keys():
        f = 'data/'+k
        return hash_get(f) == j[k]
    else:
        raise KeyError('hash for {0} not in checksums.json'.format(k))
