import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt


Quals = ['flavor_mc',
    'mouthfeel_body_mc',
    'clarity_mc',
    'aroma_mc'
    ]


def get_quals(threshold=1,brew_id = 'all', has_limits = None):
    usr2num = pd.read_pickle('../3_databases/usr2num.pkl') #brewnumber x users
    if has_limits is None:
        tas = pd.read_pickle('../3_databases/tasters.pkl')
        print 'Using all the tasting data. Shape =', tas.shape

    else:
        tas = has_limits.copy()
        print 'Using a subset of the data. Shape =', tas.shape

    if brew_id == 'all':
        print 'Finding the data for all brew ids'
        tas = tas[tas['brewnumber']<250229001] # drops bad-data rows
        tas = tas[tas['brewnumber']>140223052] # drops bad-data rows
        counts = usr2num.max(axis = 0)

    else:
        print 'Finding the data only for brew id', brew_id
        tas = tas[tas['brewnumber']==brew_id]
        usr2num = usr2num.loc[brew_id]
        counts = usr2num


    most_exp = counts[counts>threshold]
    print 'Found all the experts: Len =', len(most_exp)


    quals = []
    for quality in Quals:
        q = pd.pivot_table(tas, values= quality, columns = 'brewnumber', index = 'reg_code')
        q = q.loc[most_exp.index]
        q = q.dropna(axis = 1, how = 'all')
        q = q.dropna(axis = 0, how = 'all')
        q = q.drop_duplicates(keep='first')

        quals.append(q) # somehow they are not the same shape or have the same set of tasters
        print 'Made q matrix for {} with shape {}'.format(quality,q.shape)


    f = quals[0]
    b = quals[1]
    c = quals[2]
    a = quals[3]
    print '-------DONE-------'
    print ''
    return most_exp.reset_index().values, [f,c,a,b] # quals

# ------------------------------------------------------------------
if __name__ == '__main__':
    pass
