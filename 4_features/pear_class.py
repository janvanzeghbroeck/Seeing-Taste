import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
from quals import get_quals

Quals = ['flavor_mc',
    'mouthfeel_body_mc',
    'clarity_mc',
    'aroma_mc'
    ]
class VisualizeTasting(object):
    def __init__(self, thresh):
        ''' runs when object instantiated and crates a figure'''
        self.thresh = thresh # threshold of tastings to be considered expert
        self.quals = []
        self.ids = []
        self.tastings = []
        self.sci = []
        self.sci_big = []
        self.colors =  ['b','g','c','m','k'] # colors used to plot
        self.labels = ['Flavor','Clarity','Aroma','Body']


        plt.style.use('ggplot')

# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def fit(self,tas = None):
        '''
        INPUTS
            tas --> Type = DataFrame --> a more specific version of base fram tas aka tasters.pkl
        OUTPUTS
            None
        FUNCTION
            fits the class with the data needed to plot
        '''
        # gets the tasters(ids and tasting numbers) and quality matricies
        tasters, self.quals = get_quals(threshold=self.thresh, brew_id = 'all', has_limits = tas)
        self.ids = tasters[:,0]
        self.tastings = tasters[:,1]

        sci = pd.read_pickle('../3_databases/clean_sci.pkl')
        # gets sci to have the same brewnumbers and our quals
        sci = sci.set_index('brewnumber_a')
        i = self.quals[0].T.index # index of self.quals for brewnumber
        sci = sci.copy().loc[i]
        # cleaning further
        sci = sci.drop('flavor_a',axis = 1) # dtrops 'FT' row
        sci = sci.dropna(how = 'all') # drops any line with all na
        self.sci_big = sci.copy()


        # filling in nans with mean values for each colemn
        for col in sci.columns.tolist()[6:]:
            mean = sci[col].mean()
            self.sci_big[col] = sci[col].fillna(mean)

        sci_small = self.sci_big[['abv','ae','rdf']]
        self.sci = sci_small

        # get the same brewnumbers in both quals and sci
        i = sci_small.index

        self.quals = [q[i] for q in self.quals]







        # In [156]: i1 == i2 # i = index for quals[0] and sci
        # Out[156]: True


# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def mean_ratings(self):
        counts = []
        means = []
        for i,q in enumerate(self.quals):
            means.append(q.mean(axis = 1))
            counts.append(q.count(axis = 1))
        return counts[0], means

# ------------------------------------------------------------------
# ------------------------------------------------------------------


    def plot_pears(self):
        '''
        INPUTS
            None
        OUTPUTS
            Reuturns nothing
        FUNCTION
            Plots the distribution of the average taster scores as a violin plot
        '''
        plt.figure() # could make this a function
        # gets the means for each taster
        vio_means = [q.T.mean() for q in self.quals]
        # pltos the means
        plt.violinplot(vio_means, showmeans=True,showmedians=False)

        # sets the axis and labels
        plt.xticks([1,2,3,4],['Flavor','Clarity','Aroma','Body'],fontsize=20)
        # plt.ylabel('Higher scores = further from True to Brand',fontsize=18)
        plt.yticks([0,.2,.4,.5],['100%\nTTB','80%\nTTB', '60%\nTTB',''],fontsize = 14)
        plt.title('Distribution of Tasters with 5 individual Tasters', fontsize = 18)
# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def plot_tasters(self, taster_codes = None, tastings = None, colors = None):
        '''
        INPUTS
            taster_codes
                --> Type = list --> prints those in the list
                --> None --> None defaults to the first five tasters of self.ids
            tastings
                --> type = list of same size as taster_codes
                --> None --> defaults to first five in self.tastings
            colors
                --> Type = list --> colors used to plots
                --> None --> defaults to self.colors
        OUTPUTS
            returns nothing
        FUNCTION
            plots the average for each taster id

        '''
        self.plot_pears()
        if colors is None:
            colors = self.colors

        if taster_codes is None:
            taster_codes = self.ids
            tastings = self.tastings

        for i in range(len(colors)):
            # only does the same amount as color options

            # --- average score for each taster for each quality
            taster_means = [q.T.mean().loc[taster_codes[i]] for q in self.quals]

            if tastings is None:
                taster_label = '{}'.format(taster_codes[i])
            else:
                taster_label = 'Taster {}: {} total tastes'.format(i,int(tastings[i]),fontsize = 16)

            plt.plot(np.arange(1,5), taster_means, color = colors[i], linestyle = '--', marker = 'd', markersize = 7, linewidth = 2, label = taster_label)

        plt.legend(loc='upper left')
        plt.savefig('../figures/tasters.png', dpi=128)


# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def plot_groups(self,k):
        '''
        INPUTS
            k --> type = int --> number of groups to create based on number of tastings
        '''
        self.plot_pears()
        i = np.argsort(self.tastings)
        i_groups = np.array_split(i, k, axis = 0)
        group_means = []
        for i,g in enumerate(i_groups):
            taster_means = []
            for id_ in self.ids[g]:
                taster_means.append([q.T.mean().loc[id_] for q in self.quals])
            group_mean = np.array(taster_means).mean(axis = 0)
            group_means.append(group_mean)

            group_min = self.tastings[g].min()
            group_max = self.tastings[g].max()
            group_label = 'Group {}: {}-{} tastings'.format(i,int(group_min),int(group_max))
            plt.plot(np.arange(1,5), group_mean, color = self.colors[i], linestyle = '--', marker = 'd', markersize = 7, linewidth = 2,label = group_label)

        plt.legend(loc='upper left')
        return group_means

# ------------------------------------------------------------------
# ------------------------------------------------------------------


    def plot_brew(self,brewnumber = None,last = 10,make_fig = True):
        if brewnumber is None:
            # sets brewnumber to most recent
            brewnumber = np.max(self.quals[0].columns.tolist())


        f, axarr = plt.subplots(4, figsize = (7,8), sharex=True, sharey = False)
        axarr[0].set_title('Average tasting score for last {} tastings'.format(last),fontsize = 18)
        labels = ['Flavor','Clarity','Aroma','Body']
        for i, label in enumerate(labels):
            q_most_recent = self.quals[i].T.iloc[-last:,:]
            most_recent_numbers = q_most_recent.index.tolist()
            avg_ttb = q_most_recent.mean(axis = 1)

            axarr[i].plot(avg_ttb.values)
            axarr[i].set_ylabel(label,fontsize = 20)

            qual_mean = self.quals[i].mean().mean()
            # axarr[i].axhline(qual_mean, alpha = .7)
            three_std = qual_mean + 3* avg_ttb.std()
            axarr[i].axhline(three_std,linestyle = '--',alpha = .7)
            axarr[i].axvline(3,color = 'k')


            axarr[i].yaxis.tick_right()
            axarr[i].set_yticks([0, three_std, three_std*1.1])
            axarr[i].set_yticklabels(['TTB','upper\nlimit'],fontsize = 14)

        # adding x tick marks
        brew_numbers = map(int,most_recent_numbers)
        plt.xticks(np.arange(last),brew_numbers, rotation = 50) #could also be 'vertical'

        plt.savefig('../figures/brews.png', dpi=128)



        return brew_numbers
# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def plot_baseline(self, brewnumber = None):
        # finds the most recent brewnumber
        if brewnumber is None: # get the most recent
            most_recent = np.max(self.quals[0].columns.tolist())
        elif len(str(brewnumber)) == 11:
            most_recent = brewnumber
        else:
            most_recent = self.quals[0].columns.tolist()[brewnumber]

        # finds the taster ids in the most recent brewnumber
        tasters = self.quals[0][most_recent].dropna().index.tolist()

        labels = ['Flavor','Clarity','Aroma','Body']

        taster_scores = [] # list to turn into a data frame
        for taster in tasters:
            for i,q in enumerate(self.quals):
                # gets the mean of each tasters scores
                taster_mean = q.T.mean().loc[taster]

                # finds their score of the most recent brewnumber
                current_score = q.loc[taster,most_recent]

                # finds the difference of the two
                diff = current_score - taster_mean

                # adds the values to the future data frame
                taster_scores.append([labels[i],taster, taster_mean, current_score, diff])

        # creates data frame for the tasters scores
        taster_d = pd.DataFrame(taster_scores)
        cols = ['qual','id','mean','score','diff']
        taster_d.columns = cols

        f, axarr = plt.subplots(4, sharex=True, sharey = True)

        for i, label in enumerate(labels):
            q = taster_d[taster_d['qual']==label]
            x = np.arange(len(q))
            axarr[i].plot(x,q['mean'])
            axarr[i].set_ylabel(label)
            all_means = self.quals[i].T.mean().mean()
            axarr[i].axhline(all_means)
            three_std = all_means + 3*self.quals[i].T.mean().std()
            axarr[i].axhline(three_std,linestyle = '--')
            axarr[i].scatter(x,q['score'])
        plt.xticks(np.arange(len(tasters)),tasters,rotation='vertical')
        axarr[0].set_title('Results of tasting brew {} and the tasters average'.format(int(most_recent)))


        return taster_d

# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def get_best(self):
        qual_top_tasters = []

        f, ax = plt.subplots(4, sharex = True, sharey = True)
        for i,q in enumerate(self.quals):
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

            # adding the counts to the top tasters
            counts = pd.DataFrame(q.T.count())
            counts.columns = ['count']
            top_tasters = top_tasters.join(counts, how = 'left')

            qual_top_tasters.append(top_tasters)

            ax[i].hist(compare_means)
            # compare_three_std = compare_means.mean()+ 3*compare_means.T.std()
            # ax[i].axvline(compare_three_std)
            ax[i].set_ylabel(self.labels[i])
        ax[0].set_title('histogram of taster mean score when compared to others')


        return qual_top_tasters


# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def add_title(self,title= None):
        if title is None:
            plt.title("Total Tasters = {} \t Threshold = {} \t Beer = Fat Tire".format(len(self.ids), self.thresh),fontsize=16)

# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def plot_sci(self,last = 10):
        sci_small = self.sci_big

        if last == 'all': # plot all of them
            last = sci_small.shape[0]


        labels = sci_small.columns.tolist()[:4] # of the first four
        print_labels = ['Apparent\nExtract', 'pH', 'CO', 'ABV']
        n_cols = len(labels)
        f, axarr = plt.subplots(n_cols, figsize = (7,8), sharex=True, sharey = False)
        axarr[0].set_title('4 Chemical measures for the last {} tastings'.format(last),fontsize = 18)

        x = np.arange(last)
        sci_small = sci_small.iloc[-last:,:]

        for i, label in enumerate(labels):
            # plot change of metrics
            axarr[i].plot(x,sci_small[label])
            # set the label for each sci value
            axarr[i].set_ylabel(print_labels[i],fontsize = 20)

            # find and plot mean line (solid) and 3 std above (--)
            qual_mean = np.mean(sci_small[label])
            axarr[i].axhline(qual_mean, alpha = .7)
            three_std = 3* np.std(sci_small[label])
            axarr[i].axhline(qual_mean + three_std,linestyle = '--',alpha = .7)
            axarr[i].axhline(qual_mean - three_std,linestyle = '--',alpha = .7)

            axarr[i].axvline(3,color = 'k')

            axarr[i].yaxis.tick_right()
            # axarr[i].set_yticks([qual_mean - three_std, qual_mean, qual_mean + three_std, three_std*1.1])
            # axarr[i].set_yticklabels(['lower\nlimit','mean','upper\nlimit'],fontsize = 14)

        # setting x ticks!
        plt.xticks(x,map(int,sci_small.index.tolist()), rotation = 50)
        plt.savefig('../figures/sci.png', dpi=128)


        return sci_small



    def sci_feat(self,std_thresh = 1):
        sci = self.sci # cols = sci data, rows = brewnumber
        sci_means = sci.mean(axis = 0) # mean of each sci data
        sci_stds = sci.std(axis = 0) # std of each sci data

        # creates threshold based on std_thresh number of stds
        upper_threshold = sci_means + std_thresh*sci_stds
        lower_threshold = sci_means - std_thresh*sci_stds

        upper_idx = sci > upper_threshold # T/F of which values are greater than upper_threshold
        lower_idx = sci > lower_threshold # T/F of which values are greater than upper_threshold

        cols = sci.columns.tolist()
        index = sci.index.tolist()

        # for each tasting qualification
        scores_per_qual = []
        for i,q in enumerate(self.quals): #note there is a .5 in q
            # for each sci data
            allscores = []
            for col in cols:
                if_peak = q.T[upper_idx[col]] # brew x taster for brewnumber that have a peak
                counts = if_peak.count(axis = 0) # counts each tater has participated in a tasting with a peak

                counts_correct = if_peak.sum(axis = 0)# number of times they said 1

                if_peak = if_peak.replace(0,-1) # punish those who didnt taste it with -1 those who did already have a 1
                did_taste = counts>0
                sums = if_peak.sum(axis = 0) # sums each taster " " " "

                # scores indicates the percent of the time this taster identifies the sci peak correctly
                scores = (sums[did_taste]/counts[did_taste] + 1)/2
                # how many times they have tasted a peak for this sci measure
                scores = pd.concat([scores,counts_correct[did_taste],counts[did_taste]], axis = 1)
                # rename columns
                scores.columns = ['perc_correct','count_correct','count']
                scores['total_occur'] = len(if_peak)


                # combine into the all scores list of len(cols)
                allscores.append(scores)
            # append the allscores into a list for the 4 quals
            scores_per_qual.append(allscores)
            #^ list (4) of list (sci.cols) of dataframes (2 cols)

        return scores_per_qual



    def show(self):
        plt.show()





if __name__ == '__main__':
    sci = pd.read_pickle('databases/clean_sci.pkl')
