import pandas as pd
import numpy as np
import pickle
import re

'''
cleans all the individual tasters data from the .dat files
creates a matrix called taster.pkl
'''
def cat_one(x,one):
    if type(x) == str:
        if x.lower() == one:
            return 1
        else:
            return 0
    else:
        return np.NAN

def clean_product(x):
    if type(x) == str:
        x = x.lower()
        if x == 'bottle' or x == 'keg' or x == 'can':
            return x
        elif x == 'krg':
            return 'keg'
        elif x == 'botttle' or x == 'bptt;e':
            return 'bottle'
        else:
            return np.NAN
    else:
        return np.NAN


if __name__ == '__main__':

    df_all = pd.read_csv('../1_raw_data/ttb_dats.csv').drop('Unnamed: 0',axis = 1)
    df = df_all.copy()

    # 65936 x 59
    cols = [
        'AgeDays', # nan, int
         'AgeMonths', # nan 0,2 (count 52)
         'Aroma Comment', # ~ = nan
         'Aroma MC', # 1 = ttb , 2 = not ttb, nan
         'BatchId', # 2 rows contain brew id #
         'BestByDate', #'MM/DD/YY' nan
         'BrandCode', # 93 different name/code name
         'BrewNumber',# 2345 unique brew id
         'Clarity Comment', # ~ = nan
         'Clarity MC', # 1 = ttb , 2 = not ttb, nan
         'DateCreated',# 2 rows 'PackageDate'
         'Dept', # 33 different codes
         'Flavor Comment', # ~ = nan
         'Flavor MC',# 1 = ttb , 2 = not ttb, nan
         'FlavorName', # 2 rows 'BrandCode'
         'LotNum', #2 rows brew id
         'Mouthfeel Body MC',# 1 = ttb , 2 = not ttb, nan
         'Mouthfeel Comments',# copy of 'Mouthfeel Comments'
         'Mouthfeel comments', # ~ = nan
         'Operation', # 2 rows can 'Package'
         'Package', #['bottle' 'keg' 'BOTTLE' 'IP' 'KEG' 'can' 'CAN' 'KRG' nan]
         'PackageDate', #'MM/DD/YY' nan
         'Product Category', #'FT'
         'Proj-Name',# '2B PR NOV 10 16' code with date and packaging
         'long_one', # own tables 100 different
         'Q#1', # 10 rows
         'RecipeName', # 2 rows 'Ranger'
         'Reg-Code', # taster ID code 130 unique
         'Reg-Name', # taster Name needs .lower()
         'RepNum', # same as rep number all 1.
         'RepNumber', # [  1.  nan   7.   2.]  ???
         'RoomTempDays', # 0 or nan
         'Samp#',# [  1.   2.   3.   4.  nan   5.   6.]
         'Samp-BC', # 126 unique 3 digit ints
         'Samp-Code',# '2B 1 day bottle' similar to product name
         'Samp-Desc', # beer name
         'Samp-Pos', # [  1.   2.   3.   4.  nan   5.   6.]
         'Samp-Set', # int 1-30
         'Session', # 2 row 1.
         'SessionDate', #'MM/DD/YY'
         'Shift', # sun or star needs lower()
         'Status', # T,V needs .lower()
         'TestType', #['PR' 'IP' nan 'T' 'S']
         'Time-Stamp', # '11-10-2016 09:59:48'
         'True to Target', # comments
         'Visual Comment', # comments
         'WarmStoreDays', # 0 or nan
         '[~A000]', # incorrect upload not sure...
         'aroma mc', # same as
         'clarity mc', # same as
         'department', # same as 'Dept'
         'flavor mc', # same as
         'fresh ttb', # same as 'True to Target'
         'fresh ttb or not', # 1,2,nan
         'mouthfeel body mc', # same as
         'not fresh  ttb', # same as 'not fresh ttb' (below)
         'not fresh ttb', # ~ = nan comments
         'not true to target', # same as 'not fresh ttb'
         'true to target or not' # same as 'fresh ttb or not'
         ]


# ------------------- deleting cols
    # removes the 2 rows that have different col names
    df = df.drop([48739,48740])

    # drops the 2-row cols with no values
    df = df.dropna(axis = 1, how = 'all')

    # drops strange cols
    df = df.drop(['long_one', 'Q#1', '[~A000]', 'WarmStoreDays', 'AgeMonths', 'RepNumber', 'RepNum'],axis = 1)



# --------- combining cols with different names

    # for comment cols replace ~ with nan
    df.replace(to_replace='~', value=np.NAN, inplace=True)

    # dict of cols that are the same
    cols_diff = {'Dept':'department',
            'Mouthfeel Comments':'Mouthfeel comments',
            'Aroma MC':'aroma mc',
            'Clarity MC':'clarity mc',
            'Flavor MC':'flavor mc',
            'BrandCode':'Product Category',
            'Clarity Comment':'Visual Comment',
            'Mouthfeel Body MC':'mouthfeel body mc',
            'true to target or not':'fresh ttb or not', # 1,2,nan
            }

    for k,v in cols_diff.iteritems():
        df[k].fillna(df[v], inplace=True)
        df = df.drop(v, axis =1)

    # these were done manually because there were so many
    df['True to Target'].fillna(df['fresh ttb'], inplace=True)
    df['True to Target'].fillna(df['not fresh  ttb'], inplace=True)
    df['True to Target'].fillna(df['not fresh ttb'], inplace=True)
    df['True to Target'].fillna(df['not true to target'], inplace=True)

    df = df.drop(['fresh ttb','not fresh  ttb','not fresh ttb','not true to target'],axis = 1)



# ------------- renaming cols
    cols = df.columns.tolist()
    cols[-1] = 'ttb_mc' # was 'true to target or not'
    cols[-3] = 'ttb_comments'
    cols = [re.sub('-','_',re.sub(' ', '_', col.lower())) for col in cols ]
    df.columns = cols


# ------------ drop rows with all nans
    df = df.dropna(axis = 0, how = 'all')


# --------- replacing 1/2 to 0/1
    mc_cols = ['aroma_mc',
                'clarity_mc',
                'flavor_mc',
                'mouthfeel_body_mc',
                'ttb_mc'
                ]
    for col in mc_cols:
        df[col] = df[col]-1

# ----------- turning the categories into 1/0
    # minority class is always 1
    df['status'] = df['status'].map(lambda x: cat_one(x,'v'))  # v = valid = 1 / t = training = 0
    df['shift'] = df['shift'].map(lambda x: cat_one(x,'star')) # star = 0 / sun = 1
    df['package'] = df['package'].map(clean_product)
# --------------- printing options

    print df.info()

    # ------- prints how many unique values and the top 30

    # for col in df.columns.tolist():
    #     num = len(df[col].unique())
    #     print col,'-->', num
    #     print df[col].unique()[:30]
    #     print ''
    #     s = raw_input('--> ')

    pickle.dump( df, open( '../3_databases/tasters_2.pkl', "wb" ) )
    print 'done pickling'
