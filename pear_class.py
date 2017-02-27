import pandas as pd
import numpy as np
import pickle
import matplotlib.pylab as plt
from plots import get_quals

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
        self.labels = ['Flavor','Clairity','Aroma','Body']


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

        sci = pd.read_pickle('databases/clean_sci.pkl')
        # gets sci to have the same brewnumbers and our quals
        sci = sci.set_index('brewnumber_a')
        i = self.quals[0].T.index # index of self.quals for brewnumber
        sci = sci.copy().loc[i]
        # cleaning further
        sci = sci.drop('flavor_a',axis = 1) # dtrops 'FT' row
        sci = sci.dropna(how = 'all') # drops any line with all na
        self.sci_big = sci.copy()

        sci_small = sci.iloc[:,:5]
        # get the same brewnumbers in both quals and sci
        i = sci_small.index

        self.quals = [q[i] for q in self.quals]
        self.sci = sci_small

        # filling in nans with mean values for each colemn
        for col in sci.columns.tolist()[6:]:
            mean = sci[col].mean()
            self.sci_big[col] = sci[col].fillna(mean)




        # In [156]: i1 == i2 # i = index for quals[0] and sci
        # Out[156]: True


# ------------------------------------------------------------------
# ------------------------------------------------------------------
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
        plt.xticks([1,2,3,4],['Flavor','Clairity','Aroma','Body'],fontsize=16)
        plt.ylabel('Average Tasting Score: 0 = True- / 1 = Off-Brand',fontsize=16)
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
                taster_label = '{}: {} total tastes'.format(taster_codes[i],int(tastings[i]))

            plt.plot(np.arange(1,5), taster_means, color = colors[i], linestyle = '--', marker = 'd', markersize = 7, linewidth = 2, label = taster_label)
        plt.legend(loc='upper left')

# ------------------------------------------------------------------
# ------------------------------------------------------------------

    def plot_groups(self,k):
        '''
        INPUTS
            k --> type = int --> number of groups to create based on number of tastings
        '''
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


    def plot_brew(self,brewnumber = None,last = 10):
        if brewnumber is None:
            # sets brewnumber to most recent
            brewnumber = np.max(self.quals[0].columns.tolist())

        last_brews_lst = self.quals[0].columns.tolist()
        last_brews_lst.sort()
        last_brews = last_brews_lst[-last:]
        last_brews_values = []
        for q in self.quals:
            dist = []
            for brew in last_brews:
                dist.append(np.mean(q[brew].dropna().values))
            last_brews_values.append(dist)

        f, axarr = plt.subplots(4, sharex=True, sharey = True)
        axarr[0].set_title('Average score for last {} brewnumbers'.format(last))
        labels = ['Flavor','Clairity','Aroma','Body']
        for i, label in enumerate(labels):
            axarr[i].plot(last_brews_values[i][::-1])
            axarr[i].set_ylabel(label)
            qual_mean = np.mean(last_brews_values[i])
            axarr[i].axhline(qual_mean, alpha = .7)
            three_std = qual_mean + 3* np.std(last_brews_values[i])
            axarr[i].axhline(three_std,linestyle = '--',alpha = .7)

        # adding x tick marks
        plt.xticks(np.arange(len(last_brews)),map(int,last_brews), rotation = 70) #could also be 'vertical'



        return last_brews_values
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

        labels = ['Flavor','Clairity','Aroma','Body']

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
        sci_small = self.sci

        if last == 'all': # plot all of them
            last = sci_small.shape[0]

        n_cols = sci_small.shape[1]
        f, axarr = plt.subplots(n_cols, figsize = (7,8), sharex=True, sharey = False)
        axarr[0].set_title('Science data for last {} brewnumbers'.format(last))

        labels = sci_small.columns.tolist()
        x = np.arange(last)
        sci_small = sci_small.iloc[-last:,:]

        for i, label in enumerate(labels):
            # plot change of metrics
            axarr[i].plot(x,sci_small[label])
            # set the label for each sci value
            axarr[i].set_ylabel(label)

            # find and plot mean line (solid) and 3 std above (--)
            qual_mean = np.mean(sci_small[label])
            axarr[i].axhline(qual_mean, alpha = .7)
            three_std = 3* np.std(sci_small[label])
            axarr[i].axhline(qual_mean + three_std,linestyle = '--',alpha = .7)
            axarr[i].axhline(qual_mean - three_std,linestyle = '--',alpha = .7)


        # setting x ticks!
        plt.xticks(x,map(int,sci_small.index.tolist()), rotation = 50)

        return sci_small



    def show(self):
        plt.show()





if __name__ == '__main__':
    sci = pd.read_pickle('databases/clean_sci.pkl')
