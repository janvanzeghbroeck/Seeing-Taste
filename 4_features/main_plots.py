
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
from quals import get_quals
from pear_class import VisualizeTasting
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sil_plots import sil_plot
from taster_kmeans import TastingGroups, scree_plot

# -------------------- below is a copy from ind.py pca week 6 dim-reduct


def make_names(end_labels):
    feat_names_all = []
    for q in labels:
        feat_names = []
        for s in end_labels:
            feat_names.append("{}_{}".format(q,s))
        feat_names.append('counts')
        feat_names_all.append(feat_names)
    return feat_names_all


def get_feats(counts,score,vt_object):
    # ------------ creating the feature matrix for each tater
    # sci df column names
    sci_labels = vt_object.sci.columns.tolist()
    labels = ['Flavor','Clarity','Aroma','Body']

    # adds the count feature to the frame

    all_feats = [] # list for featture dataframes one for each qual
    for i in range(len(labels)): # for i = num of each qualifier
        # creates empty data frame for the features
        # creates the avg and tfibf features and adds with name
        feats = pd.DataFrame()
        feats['counts'] = counts
        feats['{}_tfibf'.format(labels[i])] = best[i].iloc[:,0]
        feats['{}_avg'.format(labels[i])] = means[i]
        for j in range(len(score[i])): # for j = num of sci_cols
            # gets only the percent correct col and replaces zeros with -1 and later nans with 0
            score_feat = score[i][j]['perc_correct'].replace(0,-1)
            # adds score percent to df and add col name
            feats['{}_{}'.format(labels[i],sci_labels[j])] = score_feat
        feats = feats.fillna(0)
        # list for featture dataframes one for each qual len = 4
        all_feats.append(feats)
        # ------ ends a list with four data frames
    return all_feats


# --------- end copy from ind.py pca

Quals = ['flavor_mc',
    'mouthfeel_body_mc',
    'clarity_mc',
    'aroma_mc'
    ]
labels = ['Flavor','Clarity','Aroma','Body']

if __name__ == '__main__':

    # ------------
    plt.style.use('ggplot')
    plt.close('all')
    np.random.seed(42)

    # ------------
    tas = pd.read_pickle('../3_databases/tasters.pkl')
    df = tas.copy()
    # df = df[df['status'] == 1.0]
    ft = df[df['brandcode'] == 'FT'] #limit to fat tire
    ft_pr = ft[ft['testtype'] == 'PR'] # limit to Product review
    sp = df[df['testtype'] == 'S'] # limit to spiked review
    t = df[df['testtype'] == 'T']
    # ------------

    thresh = 13

    # tasters, quals = get_quals(threshold = thresh, has_limits = ft_pr) #threshold = tasters with more than that many tastings aka experts


    vt = VisualizeTasting(thresh)
    vt.fit(ft_pr)
    vt.plot_tasters()
    # vt.add_title()
    brew = vt.plot_brew(last = 19)
    # taster_d = vt.plot_baseline(brewnumber = 161216081.0) # need XXXXXXXXX.0
    best = vt.get_best()
    sci = vt.plot_sci(last = 19)
    score = vt.sci_feat(std_thresh = 2)

    # vt.plot_groups(5)
    vt.show()
    counts, means = vt.mean_ratings()

    raise SystemExit(0) # stops the code here

    # makes the feature matrix for the kmeans
    all_feats = get_feats(counts,score,vt)


# ------------- start kmeans
    feats = all_feats[0] # just flavor
    km = TastingGroups()

    km.fit(all_feats[0])
    # km.plot_elbow(np.arange(2,30))

    # plt.close('all')

    all_names = make_names(['avg','tfibf','abv','co','ae','rdf'])
    k = 4
    for i in range(4):
        # plt.close('all')
        print '-->', labels[i], '-'*10
        km.fit(all_feats[i])
        km.plot_sil([2,8], pca_dim = 2, plot_scree = False, label = labels[i])
        km.fit_k(8, pca_dim = None)
        km.plot_features(names = all_names[i])
        plt.show()
        print ''
        s = raw_input('--> ')
