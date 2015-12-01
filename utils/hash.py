import os
import json
import pickle
import hashlib

import numpy as np


def hash_update(p, h, force_update=False):
    """Add a new filename and hash to `hashes.json`

    Parameters
    ----------
    p : str
        The filename (without a path) for a particular pickled object
    h : str
        The hash for `p`
    force_update : bool
        Whether to replace 

    Returns
    -------
    None
    """
    hashes = 'data/hashes.json'
    kv = {p : h}
    if os.path.isfile(hashes):
        with open(hashes) as f:
            data = json.load(f)

        if p in data.keys():
            if force_update:
                data[p] = h
        else:
            data.update(kv)
        with open(hashes, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(hashes, 'w') as f:
            json.dump(kv, f, indent=4)

def hash_get(ppath):
    """Get the MD5 hash based on the rounded sum of the matrix values
    for a specific pickled object.

    Parameters
    ---------
    ppath : str
        The (relative path) filename for the particular object

    Returns
    -------
    m.hexdigest: str
        The digest containing only hexadecimal digits
    """
    X = pickle.load(open(ppath, 'rb'))
    if ppath == 'data/datamatrix.pkl':
        X = round(X.sum(), 8)
    elif ppath == 'data/pca_dict_50.pkl':
        X = np.round(X['X_reduced'][0].sum(), 8)
    # elif ppath == 'data/km_dict.pkl':
        # pass
    else:
        raise ValueError('not a recognized .pkl file')
    m = hashlib.md5()
    m.update(X)
    return m.hexdigest()
    
def hash_same(p):
    """Get the hash and check it against what's in checksums.json.

    Parameters
    ----------
    p : str
        The filename (without a path) for the particular object

    Returns
    -------
    valid : bool
        True if the hashes match, False otherwise
    """
    j = json.load(open('data/hashes.json', 'r'))
    if p in j.keys():
        f = 'data/'+p
        return hash_get(f) == j[p]
    else:
        raise KeyError('hash for {0} not in checksums.json'.format(p))

def script_updated(ppath, spath):
    """To determine whether the script has been updated since the
    pickled object was created. Assumes that a more recent timestamp
    corresponds to a file whose changes we'll want to include.

    Parameters
    ----------
    ppath : str
        The (relative path) filename for the pickled object
    spath : str
        The (relative path) filename for the script

    Returns
    -------
    valid : bool
        True is script more recent than pickle, False otherwise
    """
    pt = os.path.getmtime(ppath)
    st = os.path.getmtime(spath)
    return st > pt

def make(fn, pname, sname, kwargs):
    """Wrapper function for determining whether to create or load
    a pickled object

    Parameters
    ----------
    fn : function
        A function such as `run_PCA` that deals with a pickled object
    pname : str
        The filename (without a path) for the object associated with `fn`
    sname : str
        The filename (without a path) for the script associated with `fn`
    kwargs : dict
        Keyword arguments of variable length

    Returns
    -------
    Pickled object by either creating or loading it
    """
    ppath = 'data/'+pname
    spath = 'utils/'+sname
    if os.path.isfile(ppath):
        if hash_same(pname):
            if script_updated(ppath, spath):
                fn(pname, **kwargs)
                pass
            else:
                X = pickle.load(open(ppath, 'rb'))
                return X
        else:
            fn(pname, **kwargs)
    else:
        fn(pname, **kwargs)
