import pandas as pd
import numpy as np
import pickle
import re

def add_new(filename, save = True):

    '''

    loads a new .DAT file and adds it to the quals matrix

    the file needs to be .DAT that is formatted correctly

    below is only for PR tastings

    Will update brewnumber if it already exists
    will allow for multiple brewnumbers in a file
    Will automatically filter out those who are valid

    below is an example of a file that is
    'FT CAN PR FEB 1 17.DAT'
    '''

    new = pd.read_table(filename)

    cols = new.columns.tolist()
    cols = [ col.lower() for col in cols]
    new.columns = cols

    interest = ['reg-code','flavor mc', 'clarity mc', 'aroma mc', 'mouthfeel body mc']

    df = new[interest].set_index('reg-code').apply(lambda x: x-1)
    df['status'] = new['status'].values
    df['brewnumber'] = new['brewnumber'].values

    T = df[df['status']=='T'] # training tasters
    V = df[df['status']=='V'] # valid tasters
    tasters = V.index.tolist()

    brews = new['brewnumber'].unique()

    sub_cols = interest[1:]

    quals = list(pickle.load(open('../3_databases/quals.pkl','rb')))

    new_quals = []
    new_frame = pd.DataFrame()
    for i,col in enumerate(sub_cols):
        for brew in brews:
            new_frame[brew] = V[V['brewnumber']==brew][col]
        new_quals.append(pd.concat([quals[i], new_frame], axis=1))


    # --------- printing and pickling

    print 'quals was shape:', quals[0].shape
    print 'quals now shape:', new_quals[0].shape

    if save: # == True (default) if statement to actually pickle
        pickle.dump( new_quals, open( '../3_databases/quals.pkl', "wb" ) )
        print 'Successfullly pickled new quals file'
    print 'New Brewnumbers:'
    for brew in brews:
        print brew
    return tasters, new_quals


if __name__ == '__main__':

    filename = 'FT CAN PR FEB 1 17.DAT'
    tasters, quals = add_new(filename, save = False)
