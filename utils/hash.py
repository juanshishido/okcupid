import os
import json
import pickle
import hashlib

import numpy as np


def hash_update(p, h, force_update=False):
    """For adding or updating keys (filenames) and values (hashes)
    in `hashes.json`. If the key already exists, only update
    if `force_update == True`. If it doesn't exist, add it.

    Parameters
    ----------
    p : str
        The pickled object filename
    h : str
        The hash for `p`
    force_update : bool
        True to replace hash in `hashes.json`, False otherwise

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

def hash_get(p):
    """Get the MD5 hash for `p` based on the rounded sum of the
    matrix values. Only include filename; path will be assumed.
    Use control flow for handling differences in the pickled
    object structures.

    Parameters
    ---------
    p : str
        The pickled object filename

    Returns
    -------
    m.hexdigest: str
        The digest containing only hexadecimal digits
    """
    ppath = 'data/'+p
    X = pickle.load(open(ppath, 'rb'))
    if ppath == 'data/datamatrix.pkl':
        X = round(X.sum(), 8)
    elif ppath == 'data/pca_dict_50.pkl':
        X = np.round(np.abs(X['X_reduced']).sum(), 1)
    # elif ppath == 'data/km_dict.pkl':
        # pass
    m = hashlib.md5()
    m.update(X)
    return m.hexdigest()
    
def hash_same(p):
    """Get the hash associated with `p` and compare to what's in
    `hashes.json`. Only include filename; path will be assumed.

    Parameters
    ----------
    p : str
        The pickled object filename

    Returns
    -------
    valid : bool
        True if the hashes match, False otherwise
    """
    j = json.load(open('data/hashes.json', 'r'))
    if p in j.keys():
        return hash_get(p) == j[p]
    else:
        raise KeyError('hash for {0} not in checksums.json'.format(p))

def script_updated(p, s):
    """To determine whether the script has been updated since the
    pickled object was created. Assumes that a more recent timestamp
    corresponds to a file whose changes we'll want to include. Only
    include filenames; paths will be assumed.

    Parameters
    ----------
    p : str
        The pickled object filename
    s : str
        The script filename

    Returns
    -------
    valid : bool
        True is script more recent than pickle, False otherwise
    """
    pt = os.path.getmtime('data/'+p)
    st = os.path.getmtime('utils/'+s)
    return st > pt

def make(fn, s, kwargs):
    """Wrapper function for returning a pickled object by either
    creating or loading it.

    Parameters
    ----------
    fn : function
        A function such as `run_PCA` that creates a pickled object
    s : str
        The script filename
    kwargs : dict
        Keyword arguments of variable length
        Must include the pickled object filename `p`

    Returns
    -------
    Pickled object by either creating or loading it
    """
    assert 'p' in kwargs.keys(), '`kwargs` must include the pickled '+\
        'object filename `p`'

    p = kwargs['p']
    ppath = 'data/'+p
    pk = {'ppath' : ppath}
    kwargs.update(pk)

    if os.path.isfile(ppath):
        print('{0} exists'.format(p))
        if 'force_update' in kwargs.keys():
            print('user forcing update of {0}'.format(p))
            return fn(**kwargs)
        if hash_same(p):
            print('the hash for {0} matches what\'s in hashes.json'.format(p))
            print('checking whether the script has been updated...')
            if script_updated(p, s):
                print('it has; recreating {0}'.format(p))
                fa = {'force_update' : True}
                kwargs.update(fa)
                return fn(**kwargs)
            else:
                print('it hasn\'t; loading {0}'.format(p))
                X = pickle.load(open(ppath, 'rb'))
                return X
        else:
            print('the hash for {0} does not match what\'s in '+\
                  'hashes.json; recreating it'.format(p))
            fa = {'force_update' : True}
            kwargs.update(fa)
            return fn(**kwargs)
    else:
        print('{0} does not exist; creating it'.format(p))
        return fn(**kwargs)
