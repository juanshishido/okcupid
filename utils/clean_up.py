import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def clean_up(input_df, col_names, min_words=5):
    '''
    Input : data frame and list of columns to clean up
    Returns: cleaned data frame (overwrites those columns)
    Drops user if any essay has < min_words number of words (default = 5)
    '''
    assert isinstance(col_names, list), 'Must be type list'
    assert isinstance(input_df, pd.DataFrame), 'Must be pd.DataFrame'
    dfs = []
    for c in col_names:
        df = input_df.copy()
        df[c] = df[c].replace(np.nan, '' , regex=True) \
                     .apply(lambda x: BeautifulSoup(x).getText().replace('\n', ' '))\
                     .apply(lambda x: re.sub(r"(?:\@|https?\://)\S+", "", x))\
                     .apply(lambda x: re.sub('[.]{2,}', '. ', x))\
                     .apply(lambda x: re.sub('[-]{2,}', ' - ', x))\
                     .apply(lambda x: re.sub('\s+', ' ', x).strip())
        token_count = df[c].str.split().str.len() 
        df = df[token_count > min_words] #drop rows where current essay has < min_words
        df.fillna('', inplace=True)
        dfs.append(df)
    if len(col_names) == 1:
        return dfs[0]
    else:
        return tuple(dfs)
