import pandas as pd
import numpy as np
import pickle


'''
reads taster.pkl
creates a matrix of brewnumber x users with values being their individual tasting number
brewnumbers where a user in not apart of are nan
this matrix is called usr2bru.plk
'''

if __name__ == '__main__':

    # ------------------- setting up a simple data frame: df
    DF = pd.read_pickle('../3_databases/tasters.pkl')
    df = DF.copy()
    df = df[df['brewnumber']<250229001] # drops bad-data rows
    df = df[df['brewnumber']>140223052] # drops bad-data rows


    cols_wanted = ['brewnumber',
                    'reg_code',
                    'flavor_mc',
                    'mouthfeel_body_mc',
                    'clarity_mc',
                    'aroma_mc'
                    ]

    ys = ['flavor_mc',
        'mouthfeel_body_mc',
        'clarity_mc',
        'aroma_mc'
        ]

    df = df[cols_wanted]
    df = df[df['brewnumber'].notnull()]
    df['brewnumber'] = df['brewnumber'].apply(int)

    # --------- brew2user: if a taster participated in a brewnumber = 1. not participated = 0
    brew2user_nan = pd.pivot_table(df, values= 'clarity_mc', columns = 'brewnumber', index = 'reg_code').replace(0.0,1.0)
    brew2user = brew2user_nan.fillna(0)

    # changing every value from float to int
    for col in brew2user.columns.tolist():
        brew2user[col] = brew2user[col].apply(int)


    # ------------------------------------
    tasters = df['reg_code'].unique()

    # --------finding brewnumber number for each taster
    user2num = pd.DataFrame()
    user2num['brewnumber'] = brew2user.columns.tolist()

    for taster in tasters:
        i_tasted = brew2user_nan.T[taster].notnull() #finds the index where a user participated

        user_df = brew2user.T[taster][i_tasted].reset_index()
        user_df[taster] = np.arange(1,len(user_df)+1)
        user2num = pd.merge(user2num, user_df, on='brewnumber', how='left')
        print taster

    user2num = user2num.set_index('brewnumber')
    print user2num.head()


    threshold = 100
    counts = user2num.max(axis = 0)
    most_exp = counts[counts>threshold]

    pickle.dump( user2num, open( '../3_databases/usr2num.pkl', "wb" ) )
    print 'done pickling'
