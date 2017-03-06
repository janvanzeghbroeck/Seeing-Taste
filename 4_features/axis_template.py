from pylab import *
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
from pear_class import VisualizeTasting
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

def make_plot():
    x = np.linspace(3,90,10)
    y = np.linspace(2,7,10)
    plt.scatter(x,y)

if __name__ == '__main__':


    # -------------------------------

    plt.close('all')
    plt.figure(figsize = (15,8))
    G = gridspec.GridSpec(1, 2)

    axes_1 = subplot(G[:,0])
    xticks([]), yticks([])
    img=mpimg.imread('../figures/brews.png')
    imgplot = plt.imshow(img)

    text(0.5,0.5, '',ha='center',va='center',size=24,alpha=.5)

    axes_2 = subplot(G[:,1])
    xticks([]), yticks([])
    img=mpimg.imread('../figures/sci.png')
    imgplot = plt.imshow(img)

    text(0.5,0.5, '',ha='center',va='center',size=24,alpha=.5)

    # axes_3 = subplot(G[1:, -1])
    # xticks([]), yticks([])
    # text(0.5,0.5, 'Axes 3',ha='center',va='center',size=24,alpha=.5)
    #
    # axes_4 = subplot(G[-1,0])
    # xticks([]), yticks([])
    # text(0.5,0.5, '',ha='center',va='center',size=24,alpha=.5)
    #
    # axes_5 = subplot(G[-1,-2])
    # make_plot()
    # # xticks([]), yticks([])
    # text(0.5,0.5, '',ha='center',va='center',size=24,alpha=.5)

    #plt.savefig('../figures/gridspec.png', dpi=64)
    show()
