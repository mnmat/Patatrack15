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


class Preprocessor():
    def __init__(self,path):
        self.feature_names = [
        'cp_energy', 'tkx_energy', 'cp_tkx_energy_frac', 'tkx_numtkx', 
        'weighted_bar_x', 'weighted_bar_y', 'weighted_bar_z',
        'cee_120', 'cee_200', 'cee_300', 'ceh_120', 'ceh_200', 'ceh_300', 'ceh_scint']
        
        self.initializeFeatures()
        self.fillFeatures(path)

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
    
    def fillFeatures(self,path):
        for root, dirs, files in os.walk(path):
            for file in files:
                f = uproot.open(os.path.join(path,file))
                energy = np.array([ak.sum(x) for x in f["ticlDumper/trackstersMerged"]["raw_energy"].array()])
                cp_energy = np.array([ak.sum(x) for x in f["ticlDumper/simtrackstersCP"]["regressed_energy"].array()])

                if (energy==0).any(): 
                    print(file, "contains faulty event!!! Excluded for now!!!")
                    continue       
                
                self.cp_energy.append(cp_energy)
                self.tkx_energy.append(energy)
                self.cp_tkx_energy_frac.append(energy/cp_energy)
                self.tkx_numtkx.append([ak.count(x) for x in f["ticlDumper/trackstersMerged"]["raw_energy"].array()])
                
                # Calculate Energy Weighted Barycenter
                bar_x = f["ticlDumper/trackstersMerged"]["barycenter_x"].array(library="np")
                bar_y = f["ticlDumper/trackstersMerged"]["barycenter_y"].array(library="np")
                bar_z = f["ticlDumper/trackstersMerged"]["barycenter_z"].array(library="np")
                weights = [np.array(x)/np.sum(np.array(x)) for x in f["ticlDumper/trackstersMerged"]["raw_energy"].array()]
                self.weighted_bar_x.append([np.sum([x*l]) for x, l in zip(weights, bar_x)])
                self.weighted_bar_y.append([np.sum([x*l]) for x, l in zip(weights, bar_y)])
                self.weighted_bar_z.append([np.sum([x*l]) for x, l in zip(weights, bar_z)])
                
                # Split Energy per Cell
                cell_types = [np.array(ak.sum(x,axis=0))/l for x, l in zip(f["ticlDumper/trackstersMerged"]["raw_energy_perCellType"].array(), energy)]
                cell_types = np.array(cell_types)
                self.cee_120.append(cell_types[:,0])
                self.cee_200.append(cell_types[:,1])
                self.cee_300.append(cell_types[:,2])
                self.ceh_120.append(cell_types[:,3])
                self.ceh_200.append(cell_types[:,4])
                self.ceh_300.append(cell_types[:,5])
                self.ceh_scint.append(cell_types[:,6])
        
        
        for feature_name in self.feature_names:
            setattr(self, feature_name, np.array(getattr(self, feature_name)).flatten())