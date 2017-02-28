import pandas as pd
import numpy as np
import string
import re
import matplotlib.pylab as plt
import pickle



def find_brew_num(a):
    i = 0
    while i in range(len(a)):
        brew_num = []
        while a[i] in string.digits:
            brew_num.append(a[i])
            i += 1
            if len(brew_num) > 8:
                return ''.join(brew_num)
        i +=1

def lst_brew_num():
    ans = sci['BATCH_NAME_A'].map(lambda x: find_brew_num(x))

    ans2 = []
    for a in ans:
        try:
            ans2.append(int(a))
        except:
            ans2.append('*')
    return ans2


def to_fg(x):
    try:
        step1 = 1 + x / (258.6 - 227.1 * x /258.2)
        return (step1 -1)*1000
    except:
        return np.NAN



def to_float(x):
    try:
        return float( re.sub('\*','',x))
    except:
    # print x
        return np.NAN



if __name__ == '__main__':
    sci = pd.read_csv('NB_science.csv')

    sci['BREWNUMBER_A'] = lst_brew_num()

    # the one that did not have a 9 digit code
    # print sci[sci['BREWNUMBER_A']=='*']['ID_NUMERIC_A']

    sci.replace(to_replace='*', value=np.NAN, inplace=True)


    df_all = pd.merge(ttb, sci, on='BREWNUMBER_A', how='outer')


    df = df_all.copy()
    # --------- Dropping cols

    # --------- cleaning column names
    cols = []
    for col in df.columns.tolist():
        col = col.lower()
        col = re.sub('ft_bo_','',col)
        col = re.sub('_i','',col)
        # col = re.sub('_a','',col)
        # col = re.sub('_d','',col)
        # col = re.sub('_p','',col)
        if col == 'ea':
            col = 'ae'
        cols.append(col)

    df.columns = cols


    # ------ Organizing Features

    float_cols = ['ae',
                'ph',
                'co',
                'abv',
                'rdf',
                'abw',
                'ibu',
                't90',
                't11',
                'fo',
                'but',
                'pent',
                'dms',
                'ace',
                'eace',
                'ebut',
                'ehex',
                'isoa',
                'ialc',
                'pace',
                'tiaa',
                'a4vg',
                'isocohumulone',
                'isoadhumulone',
                'isohumulone',
                'overallfail_p']

    date_cols = ['sessiondate_d',
                 'packageddate_d',
                 'batch_start_date_d']

    other_cols = ['valid_panelists_z',
                 'comments_a',
                 'process_stage_a']

    drop_cols = ['unnamed: 16',
                 'site_a',
                 'id_numeric_a_x',
                 'id_numeric_a_y',
                 'login_date_d_x',
                 'login_date_d_y',
                 'batch_name_a',
                 'recipe_a',
                 'flavor_number_a',
                 'recipe_number_a']

    too_little = ['acetic_acid',
                 'cal',
                 'lactic_acid',
                 'titratable_acidity',
                 'malic_acid',
                 'geranyl_acetate',
                 'myrcene',
                 'a2undecanone',
                 'ahumulene',
                 'caryophyllene',
                 'limonene',
                 'linalool',
                 'citric_acid',
                 'succinic_acid',
                 'cu',
                 'fe',
                 'zn',
                 'mn',
                 'ca',
                 'k',
                 'mg',
                 'na']

    id_cols = ['nbb_brand_a',
                 'package_a',
                 'p_chart_a',]
                #  'flavor_a']

    y_cols = ['clarityfail_p',
                'aromafail_p',
                'flavorfail_p',
                'mouthfeelfail_p',
                'overallfail_p']


    for col in float_cols:
        df[col] = map(lambda x: to_float(x),df[col])

    # df = df[df['nbb_brand_a']=='Fat Tire']

    ys = pd.DataFrame()
    for col in y_cols:
        ys[col] = df.pop(col).values

    df = df.drop(too_little, axis = 1)
    df = df.drop(drop_cols, axis = 1)
    df = df.drop(other_cols, axis = 1)
    df = df.drop(date_cols, axis = 1)
    df = df.drop(id_cols, axis = 1)


    # ----------


    # ----- Engineered features creating fg

    df['fg'] = [to_fg(x) for x in df['ae']]


    # --------
    # pickle.dump( df, open( 'clear_tasters.pkl', "wb" ) )
    pickle.dump( df, open( '../3_databases/clean_sci_2.pkl', "wb" ) )
    print 'done pickling'
