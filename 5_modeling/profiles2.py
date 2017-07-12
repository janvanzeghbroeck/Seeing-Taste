import pandas as pd
import numpy as np
import cPickle as pickle
from taster_kmeans import TastingGroups
import matplotlib.pylab as plt



'''
get taster data
create brew2taster for FCABT
generate Bias Majority Exp

'''

def get_majority(quals):

    '''
    returns the majoirty vote rate for each taster
    '''
    majority_rate = []

    for i,q in enumerate(quals):
        brew_means = q.mean() # mean for each brewnumber
        compare = brew_means * q # if taster said 1 for that brew then multiply by the mean score of that brew
        # compare_non_zero =

        compare_means = []
        for id_ in compare.index.tolist():
            nanless_comp = compare.loc[id_].dropna()
            idx = np.argwhere(nanless_comp>0).flatten()
            compare_mean = nanless_comp.iloc[idx].mean()
            compare_means.append(compare_mean)
            # print np.where(compare.loc[id_])[0].mean()
        compare_means = pd.DataFrame(compare_means)
        compare_means = compare_means.fillna(0)
        # means of user
        # ^ tells us when a user says 1 what is their average score compared to when others say 1
        compare_means = compare_means.set_index(compare.index)
        top_tasters = compare_means.sort_values(0)[::-1]
        majority_rate.append(top_tasters.iloc[:,0])

    # type list of series len = 4
    return majority_rate

def get_taster_names():

    '''
    returns a pd.series with index = taster_codes and values = taster names as str. different names seperated by '/'
    - reads from 3_databases folder
    - gets more tasters than the quals gets
    '''

    tas = pd.read_pickle('../3_databases/tasters.pkl')
    tas['reg_name'] = tas['reg_name'].apply(lambda x: x.lower())
    taster_names = tas.groupby('reg_code').aggregate(lambda x: tuple(x))['reg_name']
    taster_names = taster_names.apply(lambda x: ' / '.join(list(set(x))))
    return taster_names

# ----------------------------------------------------

class TasterProfiles(object):

    '''
    updated class to profile tasters Trustworthiness compared to eachother based on their TTB ratings
    - gets features from quals for each taster upon initialtion
    -

    '''

    def __init__(self, thresh = 13):

        # this is really based on quals
        self.labels = ['Flavor','Clarity','Aroma','Body']

        # NEED TO add total to quals
        # gets the tasters(ids and tasting numbers) and quality matricies
        self.quals = list(pickle.load(open('../3_databases/quals.pkl', 'rb')))
        self.ids = self.quals[0].index.tolist()

        # create features
        experience = [q.count(axis = 1) for q in self.quals][0] #data frame w/ 1 col
        bias = [q.mean(axis = 1) for q in self.quals] # list of series len = len(quals)
        majority_rate = get_majority(self.quals) # list of series len = len(quals)

        self.feats = []
        for i in range(len(self.quals)):
            feat = pd.concat([bias[i], majority_rate[i], experience], axis=1, join='inner')
            feat.columns = ['bias', 'majority_rate', 'experience']
            self.feats.append(feat) # list of data frames len = len(quals)

    def update_feats(self,new_feats_df):
        self.feats = new_feats_df

    def kmeans(self,k):
        '''
        function to call the kmeans class TastingGroups from taster_kmeans.py
        '''
        clusters_ = []
        self.k = k  #number of clusters
        self.kmean_models = [] # list of the kmeans models for each qual
        for i in range(0,len(self.quals)):
            km = TastingGroups()
            km.fit(self.feats[i])

            # --- Plot silhouette plot & score, PCA below
            # km.plot_sil([k], pca_dim = 2, plot_scree = False, label = self.labels[i])

            km.fit_k(k, pca_dim = None)

            # creates feature matrix col names
            km.plot_features(names = ['bias', 'majority_rate', 'experience'], title = self.labels[i]) # updates all_feats with cluster col
            clusters_.append(km.feats)
            self.kmean_models.append(km)

        # create cluster dataframe
        self.clusters = pd.DataFrame()
        for i, df in enumerate(clusters_):
            self.clusters[self.labels[i]] = df['cluster']
        # self.clusters['id'] = km.feats.index.tolist() #???

    def add_cluster_labels(self,label_matrix):
        #check the size len(label) x k
        label_matrix = np.array(label_matrix)
        wanted_shape = (len(self.labels),self.k)
        if label_matrix.shape != wanted_shape:
            print 'the labels were shape {}, shape {} was expected'.format(label_matrix.shape, wanted_shape)

        labeled_clusters = pd.DataFrame() # create a new dataframe
        for i,label in enumerate(self.labels):
            # create a new column
            new_col = label+'_tag'
            labeled_clusters[new_col] = self.clusters[label]
            for j,tag in enumerate(label_matrix[i]):
                # replace all the groups with their tag
                labeled_clusters[new_col] = labeled_clusters[new_col].replace(j,tag)
        self.labeled_clusters = labeled_clusters




if __name__ == '__main__':
    plt.close('all')
    tp = TasterProfiles()
    feats = [df.copy() for df in tp.feats]
    for feat in feats:
        for col in feat.columns.tolist():
            feat[col] = feat[col].apply(lambda x: x-.1 if x==0 else x)

    # tp.update_feats(feats)


    tp.kmeans(k = 4)
    # plt.show()

    cluster_labels = [ #0 = T, 1 = G, 2 = A, 3 = R
                    [3,1,0,2], #flavor
                    [2,0,3,1], #clarity
                    [2,0,1,3], #aroma
                    [0,1,2,3]  #body
                    ]

    # cluster_labels = [ #4 = Trustworthy, 3 = Good, 2 = Average, 1 = Rebel
    #                 [1,3,4,2], #flavor
    #                 [2,4,1,3], #clarity
    #                 [2,4,3,1], #aroma
    #                 [4,3,2,1]  #body
    #                 ]

    tp.add_cluster_labels(cluster_labels)
    avg_label = tp.labeled_clusters.mean(axis = 1).sort_values()
    names = get_taster_names()
    avg_label = pd.concat([avg_label,names,tp.feats[0]['experience']],axis = 1, join = 'inner').sort_values(0)
    avg_label.columns = ['Trust','Name(s)','# Tastings']

    trust_nameless = tp.clusters.applymap(lambda x: 4-x)
    trust = trust_nameless.join(names)
    print 'final data frame = trust'
