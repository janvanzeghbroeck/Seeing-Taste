import pandas as pd
import numpy as np
import pickle
import spacy
import string

'''
%run -i comments.py
'''
if 'nlp' not in locals():
    print("Loading English Module...")
    nlp = spacy.load('en')

    df = pd.read_pickle('../3_databases/tasters.pkl')
    brewids = df['brewnumber'].dropna().unique()
    brewids = brewids[brewids>150000000]
    brewids = brewids[brewids<170000000]
    brewids.sort()
    df.set_index('brewnumber',inplace = True)


comment_lst = [
'flavor_comment',
'clarity_comment',
'aroma_comment',
'mouthfeel_comments']

all_comments = df[comment_lst].reset_index()

comments_by_brew = [all_comments[all_comments['brewnumber'] == brewid][comment_lst] for brewid in brewids]

all_num_comments = []
for comments in comments_by_brew[41:42]: #41 has a lot of comments
    num_comments = [len(comments)]
    for col in comments:
        iscomments = comments[col].notnull().values
        num_comments.append(len(iscomments[iscomments == 1]))
    all_num_comments.append(num_comments)
