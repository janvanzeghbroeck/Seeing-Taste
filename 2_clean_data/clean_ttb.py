import pandas as pd
import numpy as np
import pickle


def clean_package(x):
    x = x.lower()
    if x == 'bottle' or x == 'keg' or x == 'can':
        return x
    elif x == 'krg':
        return 'keg'
    elif x == 'botttle' or x == 'bptt;e':
        return 'bottle'
    else:
        return np.NAN

def clean_site_a(x):
    if x == 'FTC':
        return 1
    else:
        return 0

def non_nan_int(x):
    if type(x) == str:
        return int(x)
    else:
        return np.NAN


if __name__ == '__main__':

    ttb = pd.read_csv('../1_raw_data/NB_brand.csv').drop('Unnamed: 16',axis = 1)
    ttb.replace(to_replace='*', value=np.NAN, inplace=True)

    # 7275 x 16
    cols = [
        'FLAVOR_A', # beer type 181 2-3 letter codes
        'BREWNUMBER_A', # brew identifier 7051 unique
        'PACKAGE_A', # bottle, can, keg -15 others to clean
        'SESSIONDATE_D', # 'M/D/YY' 1601 unique
        'PACKAGEDDATE_D', # 'M/D/YY' 2407 unique
        'P_CHART_A', # all are 'Y'
        'CLARITYFAIL_P', # total int
        'AROMAFAIL_P', # total int
        'FLAVORFAIL_P', # total int
        'MOUTHFEELFAIL_P', # total int
        'VALID_PANELISTS_Z', # total int
        'OVERALLFAIL_P', # total int has 3 nan
        'COMMENTS_A', # string
        'SITE_A', # nan and "FTC"
        'ID_NUMERIC_A', # same nan as site_a
        'LOGIN_DATE_D' #594 non-null 'M/D/YY HH:MM'
     ]

    # keg, bottle, can, unkown = nan
    ttb['PACKAGE_A'] = ttb['PACKAGE_A'].map(clean_package)

    # Y = 1 (now all 1)
    ttb['P_CHART_A'].replace(to_replace='Y', value=1, inplace=True)

    # FTC = 1 / nan = 0
    ttb['SITE_A'] = ttb['SITE_A'].map(clean_site_a)

    # turn string value to int
    ttb['OVERALLFAIL_P'] = ttb['OVERALLFAIL_P'].map(non_nan_int)
    ttb['ID_NUMERIC_A'] = ttb['ID_NUMERIC_A'].map(non_nan_int)

# --------- printing

    # for col in ttb.columns.tolist():
    #     num = len(ttb[col].unique())
    #     print col,'-->', num
    #     print ttb[col].unique()[:30]
    #     print ''

pickle.dump( ttb, open( '../3_databases/ttb_2.pkl', "wb" ) )
print 'done pickling'
