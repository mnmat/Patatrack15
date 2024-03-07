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

def getMaxShapeVariable(f,key,argmax_idx):
    return [x[l] for x,l in zip(f["ticlDumper/tracksters"][key].array(library="np"),argmax_idx)]

def filterMaxShapeVariables(var,f_idx):
    return [i for j, i in enumerate(var) if j not in f_idx]

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
                    self.bar_x += np.delete(weighted_bar_x,f_idx).tolist() 
                    self.bar_y += np.delete(weighted_bar_y,f_idx).tolist() 
                    self.bar_z += np.delete(weighted_bar_z,f_idx).tolist()
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

                    # Energy Measures
                    tkx_energy = np.array([ak.max(x) for x in f["ticlDumper/tracksters"]["raw_energy"].array()])
                    argmax_idx = np.array([ak.argmax(x) for x in f["ticlDumper/tracksters"]["raw_energy"].array()])
                    tkx_energy = np.where(tkx_energy==None,0,tkx_energy)

                    cp_energy = np.array([ak.sum(x) for x in f["ticlDumper/simtrackstersCP"]["regressed_energy"].array()])
                    cp_tkx_energy_frac = tkx_energy/cp_energy
                    
                    # Number of Tracksters
                    tkx_numtkx =np.array([ak.count(x) for x in f["ticlDumper/tracksters"]["raw_energy"].array()])

                    # Load shape Variables
                    sv = {}
                    for key in SHAPEVARIABLENAMES:
                        sv[key] = getMaxShapeVariable(f,key,argmax_idx)
                        
                    # Split by Cell Type
                    cell_types = [np.array(k[l])/m for k,l,m in zip(f["ticlDumper/tracksters"]["raw_energy_perCellType"].array(), argmax_idx ,tkx_energy)]

                    # Filter Events
                    f1_idx = np.where(tkx_energy==0)    # Exclude Tracksters with no energies
                    f2_idx = np.where(cp_energy<10)        # Exclude CPs with energies less than 10 GeV
                    f_idx = np.union1d(f1_idx,f2_idx)   

                    # Process shape Variables
                    for key in SHAPEVARIABLENAMES:
                        sv[key] = filterMaxShapeVariables(sv[key],f_idx)
                    
                    # Write quantities

                    self.cp_energy += np.delete(cp_energy, f_idx).tolist()
                    self.tkx_energy +=np.delete(tkx_energy,f_idx).tolist() 
                    self.cp_tkx_energy_frac += np.delete(tkx_energy/cp_energy,f_idx).tolist() 
                    self.tkx_numtkx += np.delete(tkx_numtkx,f_idx).tolist()

                    for feature_name in SHAPEVARIABLENAMES:
                        v = getattr(self, feature_name)
                        setattr(self, feature_name, v + sv[feature_name])

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