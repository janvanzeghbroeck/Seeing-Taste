# Seeing Taste

### The Visualization of Beer Tasting Through the eyes of a Data Scientist

Having a consistent high quality tasting craft product is important but accurately tasting that product can be challenging. Using New Belgium tasting data and data science techniques for clustering, statistics, &amp; the powers of python, Seeing Taste uses unsupervised learning to cluster tasters and identify who's good, who has a bias, and who has a specialized pallet - all for the sake of making great craft beer.

<img src="figures/break_line.png" width=100% height=100%/>

## Table of Contents
- [Workflow](#workflow)
- [The Data](#the-data)
- [Visualization](#visualization)
- [Engineered Features](#engineered-features)
- [K-Means Clustering](#k-means-clustering)
- [What Does it Mean?](#what-does-it-mean?)
- [Next Steps](#next-steps)
- [Contact](#contact)

<img src="figures/break_line.png" width=100% height=100%/>

## Workflow

#### Figure 1: The work flow from data acquisition to answering the research question: Who is a good taster?

<img src="figures/workflow.png" width=100% height=100%/>

## The Data

New Belgium has graciously provided tasting and scientific data for me to work with. Real world data can be messy and this was no exception. Using Pandas, Regular Expressions, and some smart while loops I was able to correct typos, fill in missing values, and extract id numbers from strings.

Figure 2, (below, left) shows the tasting data for the latest 19 tasting sessions. There are 4 main beer qualities that the tasters evaluate: flavor, clarity, aroma, and body. The red line shows the average taster score where a higher value indicates that more tasters thought that quality was Not True to Brand. On the right, are 4 scientific measurements for those same 19 sessions.

The vertical black line indicates one individual session where Apparent Extract peaks just outside the acceptable range (indicated by the dashed lines). Looking at the tasting data on the left some of our tasters may have noticed this based on those who flagged Not True to Brand on flavor.

#### Figure 2 & 3: Raw tasting & sample chemical measurement data respectively

<img src="figures/brews.png" width=45% height=45%/> <img src="figures/sci.png" width=45% height=45%/>

I limited my data to those tasters who were current on their New Belgium training and to tasting on their flagship beer, Fat Tire.

<img src="figures/break_line.png" width=100% height=100%/>

## Visualization

The first step was to create a data table to link each taster with each tasting session they participated in. This allows me to quickly and easily find all the data associated with any specific taster or tasting session.

From these connections, I started visualizing the distribution of the tasters to get a better idea of where differences occur. Below is a collection of violin plots show the distribution of average taster score for each of the four tasting qualities. 5 individual taters were plotted on top.

Amazingly, this one plot houses all of the actual tasting data and from it I was able to engineer features.

#### Figure 4: Violin Plots for Taster Distributions

![Alt text](/figures/tasters.png "Taster Distribution")

<img src="figures/break_line.png" width=100% height=100%/>

## Engineering Features

Because the tasting data only consisted of ones, zeros, and NaNs engineering features proved very important.

- __Tasting Bias__
    - High value indicates tasters who __often__ flag as Not True to Brand
    - Low value indicates tasters who __rarely__ flag as Not True to Brand
- __Majority Vote Rate__
    - High value indicates tasters who often flag as Not True to Brand when others agree
- __Chemical Sensitivity__
    - Looking back at figure 2 & 3 some spikes in chemical measures can be linked to some tasters indicating Not True to Brand. This feature attempts to reward those who do
    - High value indicates tasters who often flag a beer when measure is out of normal range
    - Score of 0 indicates they have not experienced any such spikes
    - Negative value indicates a taster who has never flagged a beer as Not True to Brand when the chemical measure spikes
- __Experience__
    - Indicates how many Fat Tire tastings the tasters have participated in


<img src="figures/break_line.png" width=100% height=100%/>

## K-Means Clustering

I used K-Mean to cluster the tasters into groups. Simply put, K-Means using a distance metric to group tasters by how similar they are to each other. The algorithm takes all features into account creating clusters that would be challenging for a human to balance. K-Means requires input of the number of clusters as a hyper-parameter.

Below on the left in Figure 5, you can see the silhouette score of the clusters. This score compares the distance of each point to its own cluster center over the distance to the other clusters. Maximizing this over varying numbers of clusters showed 8 groups as the optimum value.

#### Figure 5: Silhouette score plot and 2-D PCA visualization
<img src="figures/sil_plot.png" width=100% height=100%/>

Plot created from code provided by [Scikit-Learn](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py)

Above on the right, shows a 2-dimensional visualization of the clusters by using Principle Component Analysis (PCA) to reduce the dimensions to two for plotting. Looking at the plot, the clusters nicely have separation with very little overlap when only plotting the first two dimensions.

Now that we have our clusters, what do they actually mean?


<img src="figures/break_line.png" width=100% height=100%/>

## What Does it Mean?

Now that we have our clusters, plotting the distributions for each feature, for each quality (Flavor, clarity, etc.), and for each cluster will provide insight into how we did.

Below in Figure 6 & 7, we can see an example of when the clustering created distinctly recognizable groups.

#### Figure 6:
<img src="figures/trust_new.png" width=100% height=100%/>

#### Figure 7:
<img src="figures/bias_abv.png" width=100% height=100%/>


<img src="figures/break_line.png" width=100% height=100%/>

## Next Steps

<img src="figures/break_line.png" width=100% height=100%/>

## Contact

- Email --> vanzeghb@gmail.com
- [Linkedin](https://www.linkedin.com/in/janvanzeghbroeck/)
