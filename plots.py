import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt


Quals = ['flavor_mc',
    'mouthfeel_body_mc',
    'clarity_mc',
    'aroma_mc'
    ]


def get_quals(threshold=1,brew_id = 'all', has_limits = None):
    usr2num = pd.read_pickle('databases/usr2num.pkl') #brewnumber x users
    if has_limits is None:
        tas = pd.read_pickle('databases/tasters.pkl')
        print 'Using all the tasting data. Shape =', tas.shape

    else:
        tas = has_limits.copy()
        print 'Using a subset of the data. Shape =', tas.shape

    if brew_id == 'all':
        print 'Finding the data for all brew ids'
        tas = tas[tas['brewnumber']<250229001] # drops bad-data rows
        tas = tas[tas['brewnumber']>140223052] # drops bad-data rows
        counts = usr2num.max(axis = 0)

    else:
        print 'Finding the data only for brew id', brew_id
        tas = tas[tas['brewnumber']==brew_id]
        usr2num = usr2num.loc[brew_id]
        counts = usr2num


    most_exp = counts[counts>threshold]
    print 'Found all the experts: Len =', len(most_exp)


    quals = []
    for quality in Quals:
        q = pd.pivot_table(tas, values= quality, columns = 'brewnumber', index = 'reg_code')
        q = q.loc[most_exp.index]
        q = q.dropna(axis = 1, how = 'all')
        q = q.dropna(axis = 0, how = 'all')
        q = q.drop_duplicates(keep='first')

        quals.append(q) # somehow they are not the same shape or have the same set of tasters
        print 'Made q matrix for {} with shape {}'.format(quality,q.shape)


    f = quals[0]
    b = quals[1]
    c = quals[2]
    a = quals[3]
    print '-------DONE-------'
    print ''
    return most_exp.reset_index().values, [f,c,a,b] # quals

# ------------------------------------------------------------------
# ------------------------------------------------------------------



# ------------------------------------------------------------------
# ------------------------------------------------------------------



def avg_taster_pref(quals, taster_codes, thresh):

    '''
    INPUTS: - quals [list] --> qualities to be measures on seen above in Quals
            - taster_code [pandas.core.series.Series] --> output from get_quals

    OUTPUTS: Prints a plot for tasters comparison to
    '''

    # plotting stuff
    plt.figure()
    colors = ['b','g','c','m','k']

    # means of all the data for each quality for the violinplot
    vio_means = [q.T.mean() for q in quals]
    plt.violinplot(vio_means, showmeans=True,showmedians=False)

    for i,taster_code in enumerate(taster_codes.index[:len(colors)]):
        # only does the same amount as color options

        # --- average score for each taster for each quality
        taster_means = [q.T.mean().loc[taster_code] for q in quals]
        plt.plot(np.arange(1,5),taster_means, color = colors[i], linestyle = '--', marker = 'd',markersize = 7, linewidth = 2, label = '{}: {} total tastes'.format(taster_code,int(taster_codes[i])))


    # plot labels and such
    plt.ylabel('Average Tasting Score: 0 = True- / 1 = Off-Brand',fontsize=16)
    plt.title("5 of {} Expert Taster's Pear Plots for Fat Tire".format(len(taster_codes)),fontsize=16)
    plt.xticks([1,2,3,4],['Flavor','Clairity','Aroma','Body'],fontsize=16)
    plt.legend(loc='upper left')
    plt.show()

# ------------------------------------------------------------------
# ------------------------------------------------------------------


def ind_taster_pref(quals, taster_codes):

    '''
    INPUTS: - quals [list] --> qualities to be measures on seen above in Quals
            - taster_code [pandas.core.series.Series] --> output from get_quals

    OUTPUTS: Prints a plot for tasters comparison to
    '''

    # plotting stuff
    plt.figure()
    colors = ['b','g','c','m','k','b','g','c','m','k','b','g','c','m','k','b','g','c','m','k','b','g','c','m','k']

    # means of all the data for each quality for the violinplot
    vio_means = [q.T.mean() for q in quals]
    plt.violinplot(vio_means, showmeans=True,showmedians=False)

    for i,taster_code in enumerate(taster_codes.index):
        # only does the same amount as color options

        # --- average score for each taster for each quality
        taster_means = [q.T.mean().loc[taster_code] for q in quals]
        if 1 in taster_means:

            plt.plot(np.arange(1,5),taster_means, color = colors[i], linestyle = '--', marker = 'd',markersize = 14, linewidth = 2, label = '{}: {} total tastes'.format(taster_code,int(taster_codes[i])))


    # plot labels and such
    plt.ylim(0,1.5)
    plt.ylabel('Average Tasting Score: 0 = True- / 1 = Off-Brand',fontsize=16)
    plt.title("of {} Expert Taster's (>{} tastings) Pear Plots for Fat Tire".format(len(taster_codes),thresh),fontsize=16)
    plt.xticks([1,2,3,4],['Flavor','Clairity','Aroma','Body'],fontsize=16)
    plt.legend(loc='upper left')
    plt.show()

# ------------------------------------------------------------------
# ------------------------------------------------------------------


def ind_blobs_pref(quals, taster_codes, thresh):

    '''
    INPUTS: - quals [list] --> qualities to be measures on seen above in Quals
            - taster_code [pandas.core.series.Series] --> output from get_quals

    OUTPUTS: Prints a plot for tasters comparison to
    '''

    # plotting stuff
    plt.figure()
    colors = ['b','g','c','m','k','b','g','c','m','k','b','g','c','m','k','b','g','c','m','k','b','g','c','m','k']

    # means of all the data for each quality for the violinplot
    vio_means = [q.T.mean() for q in quals]
    plt.violinplot(vio_means, showmeans=True,showmedians=False)

    taster_means =[]
    for i,taster_code in enumerate(taster_codes.index):
        # only does the same amount as color options

        # --- average score for each taster for each quality
        taster_means.append([q.T.mean().loc[taster_code] for q in quals])

        for i,q in enumerate(zip(*taster_means)):
            ones = len(np.where(np.array(q) == 1)[0])
            zeros = len(np.where(np.array(q) == 0)[0])
            plt.scatter(i+1,1,s = ones*100, color = colors[i])
            plt.scatter(i+1,0,s = zeros*100, color = colors[i])
        # plt.
        # if 1 in taster_means:
        #     plt.plot(np.arange(1,5),taster_means, color = colors[i], linestyle = '--', marker = 'd',markersize = 13, linewidth = 2, label = '{}: {} total tastes'.format(taster_code,int(taster_codes[i])))


    # plot labels and such
    plt.ylim(0,1.5)
    plt.ylabel('Average Tasting Score: 0 = True- / 1 = Off-Brand',fontsize=16)
    plt.title("of {} Expert Taster's (>{} tastings) Pear Plots for Fat Tire".format(len(taster_codes),thresh),fontsize=16)
    plt.xticks([1,2,3,4],['Flavor','Clairity','Aroma','Body'],fontsize=16)
    plt.legend(loc='upper left')
    plt.show()

# ------------------------------------------------------------------
# ------------------------------------------------------------------


if __name__ == '__main__':

    # ------------
    plt.style.use('ggplot')
    plt.close('all')
    # ------------
    tas = pd.read_pickle('databases/tasters.pkl')
    df = tas.copy()
    ft = df[df['brandcode'] == 'FT'] #limit to fat tire
    ft_pr = ft[ft['testtype'] == 'PR'] # limit to Product review
    sp = df[df['testtype'] == 'S'] # limit to spiked review
    t = df[df['testtype'] == 'T']
    # ------------

    ttb = pd.read_pickle('databases/total_ttb.pkl')

    thresh = 100
    tasters, quals = get_quals(threshold = thresh, has_limits = ft_pr) #threshold = tasters with more than that many tastings aka experts
    avg_taster_pref(quals, tasters, thresh)

    thresh = 1
    tasters, quals = get_quals(threshold = thresh, has_limits = ft_pr) #threshold = tasters with more than that many tastings aka experts
    avg_taster_pref(quals, tasters, thresh)


    # t,q = get_quals(brew_id = 170214032)
    # ind_taster_pref(q, t)


# ---------------------------------

    # df = ttb.copy()
    # for col in df.columns.tolist():
    #     num = len(df[col].unique())
    #     print col,'-->', num
    #     print df[col].unique()[:30]
    #     print ''
    #     s = raw_input('--> ')
    #
    # c = quals[1]
    # perc = c.sum()/c.count()
    # x_bar = perc.mean()
    # std = perc.std()
    # three =  x_bar + 3*std
    # sig = perc[perc>three]
