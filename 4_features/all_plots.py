
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

# -------------------- below is a copy from ind.py pca week 6 dim-reduct

def scree_plot(pca, num_shown = 9,title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure()#figsize=(5, 3), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                 ])

    for i in xrange(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                       fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, num_shown-1+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)
    else:
        plt.title('{} dim, total Variance explained = {}'.format(num_components,sum(vals)))

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()#figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    # ax.axis('off')
    # ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12}, label = str(y[i]))

    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,1.1])

    if title is not None:
        plt.title(title, fontsize=16)

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
    # a = vt.plot_brew(last = 20)
    # taster_d = vt.plot_baseline(brewnumber = 161216081.0) # need XXXXXXXXX.0
    best = vt.get_best()
    sci = vt.plot_sci(last = 20)
    score = vt.sci_feat(std_thresh = 2)
    vt.show()
    # vt.plot_groups(5)
    counts, means = vt.mean_ratings()

    # ---- creating the feature matrix for each tater
    sci_labels = vt.sci.columns.tolist()
    feats = pd.DataFrame()
    feats['counts'] = counts
    for i in range(len(vt.quals)):
        feats['{}_tfibf'.format(labels[i])] = best[i].iloc[:,0]
        feats['{}_avg'.format(labels[i])] = means[i]
        for j in range(len(score[i])):
            score_feat = score[i][j]['perc_correct'].replace(0,-1)
            feats['{}_{}'.format(labels[i],sci_labels[j])] = score_feat
    feats = feats.fillna(0)


    # raise SystemExit(0) # stops the code here

# -------------- kmeans

    standard = StandardScaler()
    scaled_feats = standard.fit_transform(feats)


    sk_rss = []
    ks = np.arange(2,50)
    for k in ks:
        km = KMeans(n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                tol=0.0001,
                precompute_distances='auto',
                verbose=0,
                random_state = 42,
                copy_x=True,
                n_jobs=1,
                algorithm='auto')



        km.fit(scaled_feats)
        pred = km.predict(scaled_feats)

        # Metric for elbow plot
        sk_rss.append(km.inertia_) # sklearn metric for rss

    # -------- plotting elbow plot
    # plt.figure()
    # plt.plot(ks,sk_rss)
    # plt.title("KMeans 'Elbow' Plot")
    # plt.ylabel('RSS')
    # plt.xlabel('k-clusters')
    # plt.show()


    # ------------------ silhouette_score plots

    range_n_clusters = [2,3,4] # same as ks
    # sil_plot(scaled_feats,range_n_clusters)

    pca = PCA(15)
    trans_feats = pca.fit_transform(scaled_feats)

    sil_plot(trans_feats,range_n_clusters)


    # ------- adding the pred to the feature dataframe
    k = 4

    pred = KMeans(k).fit_predict(trans_feats)
    feats['cluster'] = pred
    c = feats.groupby('cluster').count()['counts']
    print c
    m = feats.groupby('cluster').mean()

    scaled_feats = pd.DataFrame(scaled_feats)
    scaled_feats['cluster'] = pred
    scaled_feats.index = feats.index
    scaled_feats.columns = feats.columns
    s = scaled_feats.groupby('cluster').std()


    interesting = ['counts', 'tfibf_Aroma', 'avg_Aroma', 'Aroma_ae']
    interesting = feats.columns.tolist()[:5]

    plt.style.use('ggplot')

    end_labels = ['avg','tfibf','ae', 'ph', 'abv']# + sci_labels[:-1]
    feat_names_all = []
    for q in labels:
        feat_names = []
        for s in end_labels:
            feat_names.append("{}_{}".format(q,s))
        feat_names.append('counts')
        feat_names_all.append(feat_names)



    for names in feat_names_all:
        f, ax = plt.subplots(len(names), k, figsize = (14,9), sharey = False)
        for i, col in enumerate(names): # features
            for j in range (k): # clusters
                cluster = feats[feats['cluster']==j]
                ax[0,j].set_title('cluster {}, n = {}'.format(j, len(cluster)))
                ax[i,j].hist(cluster[col], normed = False)
                # setting the x min and max limits as the min and max for each column
                ax[i,j].set_xlim([np.min(feats[col]),np.max(feats[col])])
                # limiting y axis
                # ax[i,j].set_ylim([0,y_lims[i]])
            ax[i,0].set_ylabel(col)

    plt.show()

    # ------- pca and scree plots
    n_coms = range_n_clusters

    # for n in n_coms:
    #     pca = PCA(n_components=n,
    #             copy=True,
    #             whiten=False,
    #             svd_solver='auto',
    #             tol=0.0,
    #             iterated_power='auto',
    #             random_state=None)
    #     trans_feats = pca.fit_transform(scaled_feats)
    #     print 'Variance Explained with', n, 'dimensions =', sum(pca.explained_variance_ratio_)

    scree_plot(pca,num_shown = 9)



    plt.show()
