import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sil_plots import sil_plot


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

    ax.set_xlabel("ncipal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)
    else:
        plt.title('{} dim, total Variance explained = {}'.format(num_components,sum(vals)))

# ------------------------------

class TastingGroups(object):
    def __init__(self):
        self.feats = [] # fitted feature matix
        self.scaled_feats = [] # scaled transformation
        self.trans_feats = [] # pca transformation
        self.k = 0
        np.random.seed(42)

    def fit(self, feats):
        self.feats = feats
        standard = StandardScaler()
        self.scaled_feats = standard.fit_transform(feats)

    def plot_elbow(self,range_ks):
        sk_rss = []
        # ks = np.arange(2,50)
        for k in range_ks:
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

            km.fit(self.scaled_feats)
            pred = km.predict(self.scaled_feats)

            # Metric for elbow plot
            sk_rss.append(km.inertia_) # sklearn metric for rss

        # -------- plotting elbow plot
        plt.figure()
        plt.plot(range_ks,sk_rss)
        plt.title("KMeans 'Elbow' Plot")
        plt.ylabel('RSS')
        plt.xlabel('k-clusters')


    # ------------------ silhouette_score plots
    def plot_sil(self, range_ks, pca_dim = None, plot_scree = True, label = None):
        plt.style.use('classic')

        if pca_dim is None: # dont use pca
            sil_plot(self.scaled_feats,range_ks,label)
        else: # use pca with dimension space
            pca = PCA(pca_dim)
            trans_feats = pca.fit_transform(self.scaled_feats)
            sil_plot(trans_feats,range_ks,label)

            if plot_scree:
                scree_plot(pca)

    def fit_k(self, k, pca_dim = None):
        self.k = k
        if pca_dim is None: # dont use pca
            km = KMeans(k).fit(self.scaled_feats)
            self.pred = km.predict(self.scaled_feats)

        else: # use pca with dimension space
            pca = PCA(pca_dim)
            self.trans_feats = pca.fit_transform(self.scaled_feats)
            km = KMeans(k).fit(self.trans_feats)
            self.pred = km.predict(self.trans_feats)



    def plot_features(self,names,title = ''):
        plt.style.use('ggplot')
        feats = self.feats
        feats['cluster'] = self.pred

        f, ax = plt.subplots(len(names), self.k, figsize = (15,8), sharey = False)
        f.canvas.set_window_title(title)
        for i, col in enumerate(names): # features
            for j in range(self.k): # clusters
                cluster = feats[feats['cluster']==j]
                ax[0,j].set_title('cluster {}, n = {}'.format(j, len(cluster)))
                ax[i,j].hist(cluster[col], normed = False)
                # setting the x min and max limits as the min and max for each column
                ax[i,j].set_xlim([np.min(feats[col]),np.max(feats[col])])
                # limiting y axis
                # ax[i,j].set_ylim([0,y_lims[i]])
            ax[i,0].set_ylabel(col)



if __name__ == '__main__':

    # raise SystemExit(0) # stops the code here

    pass
