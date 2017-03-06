## Seeing Taste

# Visualization of Beer Tasting
### Through the eyes of a Data Scientist

Having a consistent high quality tasting craft product is important but accurately tasting that product can be challenging. Using data science techniques for clustering, statistics, &amp; the powers of python, Seeing Taste uses unsupervised learning to cluster tasters and identify who's good, who has a bias, and who has a specialized pallet all for the sake of making great craft beer.


## Table of Contents
- [Workflow](#workflow)
- [The Data](#the-data)
- [Visualization](#visualization)
- [Engineered Features](#engineered-features)
- [Clustering](#clustering)
- [What Does it Mean?](#what-does-it-mean?)
- [Next Steps](#next-steps)

## Workflow

### Figure 1: The work flow from data acquisition to answering the research question: Who is a good taster?

<img src="figures/workflow.png" width=100% height=100%/>

## The Data

New Belgium has graciously provided tasting and scientific data for me to work with. Real world data can be messy and this was no exception. Using Pandas, Regular Expressions, and some smart while loops I was able to correct typos, fill in missing values, and extract id numbers from strings.

Figure 2, (below, left) shows the tasting data for the latest 19 tasting sessions. There are 4 main beer qualities that the tasters evaluate: flavor, clarity, aroma, and body. The red line shows the average taster score where a higher value indicates that more tasters thought that quality was Not True to Brand. On the right, are 4 scientific measurements for those same 19 sessions.

The vertical black line indicates one individual session where Apparent Extract peaks just outside the acceptable range (indicated by the dashed lines). Looking at the tasting data on the left some of our tasters may have noticed this based on those who flagged Not True to Brand on flavor.

#### Figure 2 & 3: raw tasting and chemical measurement data respectively

<!-- <img src="figures/brews.png" width=45% height=45%/> <img src="figures/sci.png" width=45% height=45%/> -->

<img src="figures/sci_brew.png" width=100% height=100%/>

I limited my data to those tasters who were current on their New Belgium training and to tasting on their flagship beer, Fat Tire.

## Visualization

The first step was to create a data table to link each taster with each tasting session they participated in. This allows me to quickly and easily find all the data associated with any specific taster or tasting session.

From these connections, I started visualizing the distribution of the tasters to get a better idea of where differences occur. Below is a collection of violin plots show the distribution of average taster score for each of the four tasting qualities. 5 individual taters were plotted on top.

Amazingly, this one plot houses all of the actual tasting data and from it I was able to engineer features.

![Alt text](/figures/tasters.png "Taster Distribution")


## Engineered Features

Below are the four different types of features that were engineered.

- Tasting Bias
        - 🔼  Often flags as Not True to Brand
        - 🔽  Rarely flags as Not True to Brand
- Majority Vote Rate
        - 🔼  Often flag as Not True to Brand when others agree
        - Rewards tasters who agree when the beer is Not True to Brand
- Chemical Sensitivity
    - Rewards tasters who flag a beer when a chemical measurement was outside of the acceptable range
    - 🔼 more often they flag a beer when ABV is unacceptably high
    Score of 0 indicates they have not experienced any such spikes
- Experience
    - Indicates how many Fat Tire tastings they participated in

## Clustering

## What Does it Mean?

## Next Steps
