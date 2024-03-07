import uproot
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import awkward as ak
import sys
import seaborn as sns

sys.path.append("../")

from utils.preprocessing import Preprocessor

import mplhep as hep
hep.style.use(hep.style.ROOT) # For now ROOT defaults to CMS


def plot_histogram(df,key,label,path):
    fig = plt.figure()
    plt.hist(df[key],bins=20)
    plt.xlabel(r"%s"%label)        
    plt.ylabel("N")
    plt.savefig(os.path.join(path,"hist_%s.png"%key))
    plt.savefig(os.path.join(path,"hist_%s.pdf"%key))

def plot_histogram2d(df,key1, key2,label1,label2,path):
    fig = plt.figure()
    plt.hist2d(df[key1], df[key2], bins = 50)    
    plt.xlabel(r"%s"%label1)        
    plt.xlabel(r"%s"%label2) 
    plt.savefig(os.path.join(path,"hist2d_%s_%s.png"))
    plt.savefig(os.path.join(path,"hist2d_%s_%s.pdf"))

def plot_correlation(df,path):
    corr = df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(os.path.join(path,"correlation_plot.png"))
    plt.savefig(os.path.join(path,"correlation_plot.pdf"))

labels = {"cp_energy": "Energy [GeV]",
         "tkx_energy": "Energy [GeV]",
         "cp_tkx_energy_frac": "$(\sum_T E_T)/E_{CP}$",
          "tkx_numtkx" : "$N_{Tracksters}$",
          "weighted_bar_x" : "x [cm]",
          "weighted_bar_y" : "y [cm]",
          "weighted_bar_z" : "z [cm]",
          "weighted_bar_r" : "r [cm]",
          "cee_120" : "Energy Fraction (CEE 120)",
          "cee_200" : "Energy Fraction (CEE 200)",
          "cee_300" : "Energy Fraction (CEE 300)",
          "ceh_120" : "Energy Fraction (CEH 120)",
          "ceh_200" : "Energy Fraction (CEH 200)",
          "ceh_300" : "Energy Fraction (CEH 300)",
          "ceh_scint" : "Energy Fraction (CEH Scint.)"}


if __name__ == '__main__':
    # Load Data

    # TODO: Add parser
    path = "/Users/markmatthewman/Projects/Patatrack15/data"
    file = "test.pkl"
    d = Preprocessor.loadNtuple(os.path.join(path,file))
    df = pd.DataFrame(d)

    # Plot Histograms
    outdir = "/Users/markmatthewman/Projects/Patatrack15/plots"
    keys = df.keys()
    for key in keys:
        plot_histogram(df,key,labels[key],outdir)

    # Plot 2D Histograms
    df["weighted_bar_r"] = np.sqrt(df["weighted_bar_x"]**2 + df["weighted_bar_y"]**2)
    keys1 = ["weighted_bar_x","weighted_bar_r"]
    keys2 = ["weighted_bar_y","weighted_bar_z"]
    
    for k1, k2 in zip(keys1,keys2):
        plot_histogram2d(df,k1,k2,labels[k1],labels[k2],path)

    # Correlation plot
    plot_correlation(df,path)
