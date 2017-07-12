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
from new_data import add_new
from main_plots import get_feats, make_names
from email_func import send_email
import os

if __name__ == '__main__':
    # new data file to add
    filename = 'FT CAN PR FEB 1 17.DAT'

    # ----- update tasting database
    # adds to tasting session to quals file
    tasters, quals = add_new(filename, save = False) # false for testing

    # create vilusaliztion class instance
    # tasters with more tasting than thresh
    vt = VisualizeTasting(thresh = 13)

    vt.fit(tas = None) # fit the data with quals.pkl

    # ---- creating the plots
    # plot violin plot with tasters
    plt.close('all')
    vt.plot_tasters(taster_codes = None, tastings = None)

    # plots tasting and sci data for the last n brews
    n = 19
    brew = vt.plot_brew(last = n)
    sci = vt.plot_sci(last = n)

    # plt.show()

    # raise SystemExit(0) # stops the code here

    # ----- Get our feature matrix
    # creates the majoirty vote rate feature
    majority = vt.get_majoirty(yes_plot = False)
    # creates the scientific sensitivity feature
    sensitivity = vt.sci_feat(std_thresh = 2)

    # gets experience and bias features
    experience, bias = vt.mean_ratings()

    # makes the feature matrix for the kmeans
    

    # ----- kmeans clustering
    labels = ['Flavor','Clarity','Aroma','Body']
    k = 8 # number of clusters


    i = 0 # just flavor could for loop
    km = TastingGroups()
    km.fit(all_feats[i])

    # Plot silhouette plot & score, PCA below
    # km.plot_sil([k], pca_dim = 2, plot_scree = False, label = labels[i])

    km.fit_k(k, pca_dim = None)

    # creates feature matrix col names
    all_names = make_names(['avg','tfibf','abv','ae','rdf'])
    km.plot_features(names = all_names[i]) # updates all_feats with cluster col

    # --- end qual for loop
    # plt.show()

    # --- save the model as pickle
    save = False
    if save:
        pickle.dump( all_feats, open( '../3_databases/clusters.pkl', "wb" ) )

    # ---- labeling clusters
    taster_profiles = {0:'Biased Taster',
                        1:'Average Taster 1',
                        2:'New with Potential',
                        3:'Cautious Taster',
                        4:'Trustworthy Taster',
                        5:'ABV/AE Specialist',
                        6:'Average Taster 2',
                        7:'New Taster'
                        }


    #------- get names of tasters and their cluster name
    tas = pd.read_pickle('../3_databases/tasters.pkl')
    tas['reg_name'] = tas['reg_name'].apply(lambda x: x.lower())
    taster_names = tas.groupby('reg_code').aggregate(lambda x: tuple(x))['reg_name']
    taster_names = taster_names.apply(lambda x: ' / '.join(list(set(x))))

    taster_clusters = km.feats['cluster'].apply(lambda x: taster_profiles[x])

    name_and_cluster = pd.concat([taster_clusters, taster_names], axis=1).dropna().sort_values('cluster').reset_index()
    name_and_cluster = name_and_cluster[['reg_name','index','cluster']]
    print name_and_cluster.head()
    name_and_cluster.to_csv('../3_databases/names_and_clusters.csv')



    # gets the cluster for each person in the new tasting session
    # need an if NaN
    brew_tasters = all_feats[i].loc[tasters]['cluster'].apply(lambda x: taster_profiles[x]).sort_values()
    print ''
    print 'Tasters participating:'
    print brew_tasters

    tast_str = zip(brew_tasters.index.tolist(),brew_tasters.tolist())
    tast_str = [' \t=\t '.join(r) for r in tast_str]

    alert_level = 'LOW'
    # -------- sending tha automated email

    fromaddr = "vanzeghbroeck.jan@gmail.com" #"YOUR EMAIL"
    # toaddr = "smithmds@gmail.com"
    toaddr = 'vanzeghb@gmail.com' #"EMAIL ADDRESS YOU SEND TO"
    psw = os.environ['EMAIL_PSW_DOTJAN'] #get password for email
    subject = 'Automatic Tasting Update: Alert Level {}'.format(alert_level) # subject line text
    body = ' \n '.join(tast_str) # body text
    # list of files to attach
    filename_lst = ['../figures/brews.png', '../figures/sci.png', '../figures/tasters.png', '../3_databases/names_and_clusters.csv']

    send_email(fromaddr,psw,toaddr,subject,body,filename_lst)
