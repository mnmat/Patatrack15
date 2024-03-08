import pandas as pd
import numpy as np
import uproot
import os
import xgboost as xgb
import awkward as ak
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from utils import fit
#from utils import validation
import mplhep as hep
import pickle

SHAPEVARIABLENAMES = ["EV1", "EV2","EV3","eVector0_x","eVector0_y","eVector0_z","sigmaPCA1","sigmaPCA2","sigmaPCA3","barycenter_x","barycenter_y","barycenter_z"]
FEATURENAMESSUM = ['cp_energy', 'tkx_energy', 'cp_tkx_energy_frac', 'tkx_numtkx', 'barycenter_x', 'barycenter_y', 'barycenter_z','cee_120', 'cee_200', 'cee_300', 'ceh_120', 'ceh_200', 'ceh_300', 'ceh_scint']
FEATURENAMESMAX = ['cp_energy', 'tkx_energy', 'cp_tkx_energy_frac', 'tkx_numtkx','cee_120', 'cee_200', 'cee_300', 'ceh_120', 'ceh_200', 'ceh_300', 'ceh_scint'] + SHAPEVARIABLENAMES


cpVariables = ["regressed_energy","event"]
tracksterVariables = ["EV1", "EV2","EV3","eVector0_x","eVector0_y","eVector0_z","sigmaPCA1","sigmaPCA2","sigmaPCA3","barycenter_x","barycenter_y","barycenter_z", "raw_energy","event"]
energyVariables = ["raw_energy_perCellType","event"]
energiesPCT = ['cee_120', 'cee_200', 'cee_300', 'ceh_120', 'ceh_200', 'ceh_300', 'ceh_scint']


def getMaxShapeVariable(f,key,argmax_idx):
    return [x[l] for x,l in zip(f["ticlDumper/tracksters"][key].array(library="np"),argmax_idx)]

def filterMaxShapeVariables(var,f_idx):
    return [i for j, i in enumerate(var) if j not in f_idx]


def split_energies(df):
    for i in range(7):
        df_split = df[df["subsubentry"]==i][["raw_energy_perCellType","event"]]
        df_split = df_split.rename(columns={"raw_energy_perCellType":energiesPCT[i]})
        df_split = df_split.reset_index()
        if i == 0:
            df_out = df_split
        else:
            df_out[energiesPCT[i]] = df_split[energiesPCT[i]]
    return df_out


class Preprocessor():
    def __init__(self,path,mode="Max"):

        #self.fillFeaturesSummedTrackster(path)
        if mode == "Sum":
            self.feature_names = FEATURENAMESSUM
            self.initializeFeatures()
            self.fillFeaturesSummedTrackster(path)
        elif mode == "Max":
            self.feature_names = FEATURENAMESMAX
            self.initializeFeatures()
            self.fillFeaturesMaxTrackster(path)


    def saveToPickle(self, path):
        features_to_pickle = {feature_name: getattr(self, feature_name) for feature_name in self.feature_names}
        with open(path, 'wb') as handle:
            pickle.dump(features_to_pickle, handle)

    @classmethod
    def loadNtuple(cls,path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def initializeFeatures(self):
        for feature_name in self.feature_names:
            setattr(self, feature_name, [])
    
    def fillFeaturesSummedTrackster(self,path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if ".root" in file:
                    f = uproot.open(os.path.join(path,file))

                    # Energy Measures
                    tkx_energy = np.array([ak.sum(x) for x in f["ticlDumper/trackstersMerged"]["raw_energy"].array()])
                    cp_energy = np.array([ak.sum(x) for x in f["ticlDumper/simtrackstersCP"]["regressed_energy"].array()])
                    cp_tkx_energy_frac = tkx_energy/cp_energy
                    
                    # Number of Tracksters
                    tkx_numtkx =np.array([ak.count(x) for x in f["ticlDumper/trackstersMerged"]["raw_energy"].array()])

                    # Barycenter
                    bar_x = f["ticlDumper/trackstersMerged"]["barycenter_x"].array(library="np")
                    bar_y = f["ticlDumper/trackstersMerged"]["barycenter_y"].array(library="np")
                    bar_z = f["ticlDumper/trackstersMerged"]["barycenter_z"].array(library="np")
                    weights = [np.array(x)/np.sum(np.array(x)) for x in f["ticlDumper/trackstersMerged"]["raw_energy"].array()]

                    weighted_bar_x = np.array([np.sum([x*l]) for x, l in zip(weights, bar_x)])
                    weighted_bar_y = np.array([np.sum([x*l]) for x, l in zip(weights, bar_y)])
                    weighted_bar_z = np.array([np.sum([x*l]) for x, l in zip(weights, bar_z)])

                    # Split by Cell Type
                    cell_types = [np.array(ak.sum(x,axis=0))/l for x, l in zip(f["ticlDumper/trackstersMerged"]["raw_energy_perCellType"].array(), tkx_energy)]

                    # Filter Events
                    f1_idx = np.where(tkx_energy==0)    # Exclude Tracksters with no energies
                    f2_idx = np.where(cp_energy<10)        # Exclude CPs with energies less than 10 GeV
                    f_idx = np.union1d(f1_idx,f2_idx)   
                    
                    # Write quantities
                    self.cp_energy += np.delete(cp_energy, f_idx).tolist()
                    self.tkx_energy +=np.delete(tkx_energy,f_idx).tolist() 
                    self.cp_tkx_energy_frac += np.delete(tkx_energy/cp_energy,f_idx).tolist() 
                    self.tkx_numtkx += np.delete(tkx_numtkx,f_idx).tolist()
                    self.barycenter_x += np.delete(weighted_bar_x,f_idx).tolist() 
                    self.barycenter_y += np.delete(weighted_bar_y,f_idx).tolist() 
                    self.barycenter_z += np.delete(weighted_bar_z,f_idx).tolist()
                    # Split Energy per Cell
                    cell_types = [i for j, i in enumerate(cell_types) if j not in f_idx]
                    cell_types = np.array(cell_types)
                    self.cee_120 += cell_types[:,0].tolist()
                    self.cee_200 += cell_types[:,1].tolist()
                    self.cee_300 += cell_types[:,2].tolist()
                    self.ceh_120 += cell_types[:,3].tolist()
                    self.ceh_200 += cell_types[:,4].tolist()
                    self.ceh_300 += cell_types[:,5].tolist()
                    self.ceh_scint += cell_types[:,6].tolist()
                
        
        for feature_name in self.feature_names:
            setattr(self, feature_name, np.array(getattr(self, feature_name)).flatten())


    def fillFeaturesMaxTrackster(self,path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if ".root" in file:
                    f = uproot.open(os.path.join(path,file))

                    cp = f["ticlDumper/simtrackstersCP"].arrays(cpVariables)
                    tracksters = f["ticlDumper/tracksters"].arrays(tracksterVariables)
                    energiesPerCellType = f["ticlDumper/tracksters"].arrays(energyVariables)

                    df_trackster = ak.to_dataframe(tracksters)
                    df_trackster = df_trackster.reset_index()
                    df_cp = ak.to_dataframe(cp)
                    df_splitEnergies = ak.to_dataframe(energiesPerCellType)
                    df_splitEnergies = split_energies(df_splitEnergies.reset_index())

                    nclusters = df_trackster.groupby("event").count().reset_index()[["raw_energy","event"]]
                    nclusters = nclusters.rename(columns={"raw_energy":"tkx_numtkx"})
                    max_energy_index = df_trackster.groupby('event')['raw_energy'].idxmax()
                    df_trackster = df_trackster.loc[max_energy_index]
                    df_splitEnergies = df_splitEnergies.loc[max_energy_index]
                    df_merge = pd.merge(df_trackster,df_cp,how="left",on=["event"])
                    df_merge = pd.merge(df_merge,nclusters,how="left",on=["event"])
                    df_merge = pd.merge(df_merge,df_splitEnergies,how="left",on=["event"])
                    df_merge["cp_tkx_energy_frac"] = df_merge["raw_energy"]/df_merge["regressed_energy"]

                    # Write out quantities
                    df_merge = df_merge.rename(columns = {"raw_energy":"tkx_energy",
                                                          "regressed_energy": "cp_energy"})
                    df_merge = df_merge.drop(columns = ["entry","subentry","event"])
                    print(df_merge.keys())
                    for feature_name in df_merge.keys():
                        if (feature_name == "index"): continue
                        v = getattr(self, feature_name)
                        setattr(self, feature_name, v + df_merge[feature_name].to_list())

                
        
        for feature_name in self.feature_names:
            setattr(self, feature_name, np.array(getattr(self, feature_name)).flatten())