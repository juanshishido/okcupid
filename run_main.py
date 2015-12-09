import pickle
import warnings
import sys
from utils.hash import make
from utils.calculate_pmi_features import *
from utils.clean_up import *
from utils.categorize_demographics import *
from utils.reduce_dimensions import run_kmeans
from utils.nonnegative_matrix_factorization import nmf_inspect, nmf_labels
warnings.filterwarnings('ignore')

def language_categories_simple(language):
    if language == 1:
        return 'mono'
    else:
        return 'multi'

df = pd.read_csv('data/profiles.20120630.csv')

essay_list = ['essay0','essay4','essay5']
df_clean = clean_up(df, essay_list)

df_clean.fillna('', inplace=True)

df_clean['religion'] = df_clean['religion'].apply(religion_categories)
df_clean['job'] = df_clean['job'].apply(job_categories)
df_clean['drugs'] = df_clean['drugs'].apply(drug_categories)
df_clean['diet'] = df_clean['diet'].apply(diet_categories)
df_clean['body_type'] = df_clean['body_type'].apply(body_categories)
df_clean['drinks'] = df_clean['drinks'].apply(drink_categories)
df_clean['sign'] = df_clean['sign'].apply(sign_categories)
df_clean['ethnicity'] = df_clean['ethnicity'].apply(ethnicity_categories)
df_clean['pets'] = df_clean['pets'].apply(pets_categories)
df_clean['speaks'] = df_clean['speaks'].apply(language_categories)
df_clean['speaks_simple'] = df_clean['speaks'].apply(language_categories_simple)

data_dict = dict()
splits = ['asian', 'black','hispanic / latin', 'multi', 'white']
category = 'ethnicity'
essay = [0, 4, 5]

for s in splits:
    for e in essay:
        filename = '_'.join([category, s, str(e)+'.txt'])
        print(filename)
        data_dict[s] = df_clean[df_clean[category] == s]
        count_matrix, tfidf_matrix, vocab = col_to_data_matrix(data_dict[s], 'essay'+str(e))
        nmf_inspect(tfidf_matrix, vocab)
