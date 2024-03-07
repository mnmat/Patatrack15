from sklearn.model_selection import train_test_split 
import pandas as pd

def create_split(d,target = "cp_energy",drop=["cp_tkx_energy_frac","cp_energy"], frac=0.2):
    df = pd.DataFrame(d)
    train, test = train_test_split(df, test_size=frac)
    y_train = train[target]
    X_train = train.drop(drop,axis=1)
    y_test = test[target]
    X_test = test.drop(drop,axis=1)
    return X_train, y_train, X_test, y_test