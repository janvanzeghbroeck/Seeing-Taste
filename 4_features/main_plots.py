
import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
from quals import get_quals
from pear_class import VisualizeTasting

Quals = ['flavor_mc',
    'mouthfeel_body_mc',
    'clarity_mc',
    'aroma_mc'
    ]

if __name__ == '__main__':

    # ------------
    plt.style.use('ggplot')
    plt.close('all')
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
    vt.plot_pears()
    vt.plot_tasters()
    vt.add_title()
    a = vt.plot_brew(last = 20)
    taster_d = vt.plot_baseline(brewnumber = 161216081.0) # need XXXXXXXXX.0
    best = vt.get_best()
    sci = vt.plot_sci(last = 20)
    vt.show()
    # vt.plot_groups(5)

    # plt.figure()
    # vt.plot_pears()
    # vt.plot_tasters()
    # vt.add_title()
    # # vt.plot_groups(5)
    # plt.show()


    # thresh = 100
    # tasters, quals = get_quals(threshold = thresh, has_limits = ft_pr) #threshold = tasters with more than that many tastings aka experts
    # avg_taster_pref(quals, tasters, thresh)
    #
    # thresh = 1
    # tasters, quals = get_quals(threshold = thresh, has_limits = ft_pr) #threshold = tasters with more than that many tastings aka experts
    # avg_taster_pref(quals, tasters, thresh)
