# preprocessing.py
import pandas as pd 
import numpy as np 
import matplotlib as plt
from dataclasses import dataclass 
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


#column Strandralization:
def  column_stand(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ","_")
        .str.replace("-","_") 
          )
    return df

#finding mising values in dataset

def is_missing(df):
   missing = pd.DataFrame({
       "missig_count":df.isnull().sum(),
       "Missing %":round(df.isnull().mean()*100,2)

    })
   return missing.sort_values("Missing %",ascending = False )

#duplicate remover
def remove_dup(df):
    return df.drop_duplicates().reset_index(drop = True)

# feature finder
def get_numeric(df):
    return df.select_dtypes(include = np.number).columns.to_list()

def get_cat(df):
    return df.select_dtypes(include = "object").columns.to_list()

#missing value handler
def fill_num(df , strategy = "median",imputer = None):
    num_cols = get_numeric(df)
    if len(num_cols) == 0:
        return df,None
    if imputer is None:
      imputer = SimpleImputer(strategy = strategy)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df,imputer

def fill_cat(df , strategy = "most_frequent", imputer = None):
    cat_cols = get_cat(df)
    if len(cat_cols) == 0:
        return df,None
    if imputer is None:
      imputer = SimpleImputer(strategy = strategy)
    df[cat_cols] = imputer.fit_transform(df[cat_cols])
    return df,imputer

#encoders
def label_encoders(df,encoder = None):
    df = df.copy()
    encoders = {}
    for col in get_cat(df):
        if encoder is None:
          encoder = LabelEncoder()
          df[col] = encoder.fit_transform(df[col])
        else:
          df[col] = encoder.transform(df[col])
        encoders[col] = encoder
        return df,encoders
       
def one_hot(df, columns=None,encoder = None):
    df = df.copy()
    if columns is None:
        columns = get_cat(df)
    if encoder is None:
      encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
      )

      encoded = encoder.fit_transform(df[columns])
    else:
      encoded = encoder.transform(df[columns])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(columns),
        index=df.index
    )

    df = df.drop(columns=columns)
    df = pd.concat([df, encoded_df], axis=1)

    return df, encoder
   
    
#scalers
def stand_scaler(df,columns , scaler = None):
    df = df.copy()
    if scaler is None:
      scaler = StandardScaler()
      df[columns] = scaler.fit_transform(df[columns])
    else:
      df[columns] = scaler.transform(df[columns])
    return df,scaler

def minmax_scaler(df,columns,scaler = None):
    df = df.copy()
    if scaler is None:
      scaler = MinMaxScaler()
      df[columns] = scaler.fit_transform(df[columns])
    else:
      df[columns] = scaler.transform(df[columns])
    return df,scaler

def robust_scaler(df,columns,scaler = None):
    df = df.copy()
    if scaler is None:
      scaler = RobustScaler()
      df[columns] = scaler.fit_transform(df[columns])
    else:
      df[columns] = scaler.transform(df[columns])
    return df,scaler

#outlier finder
def is_outlier(df,columns=None):
    if columns is None:
        columns = get_numeric(df)
    outlier = {}
    for col in columns:
      Q1 = df[col].quantile(0.25)
      Q3 = df[col].quantile(0.75)
      IQR = Q3 - Q1
     
      lower = Q1 - 1.5 * IQR
      upper = Q3 + 1.5 * IQR
      
      mask = (df[col]< lower)|(df[col]> upper)
      outlier[col]= mask.sum()
    return outlier

#remove outliers
def remove_outlier(df, columns, bounds=None):
    df = df.copy()

    if bounds is None:
        bounds = {}

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            bounds[col] = (lower, upper)

    for col in columns:
        lower, upper = bounds[col]
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df, bounds

@dataclass
class analyzefe:
    missing_valuecount:pd.DataFrame
    numeric_features:list
    categorical_features:list
    outliers_detection:dict

def analyze(df):
    missing = is_missing(df)
    numeric = get_numeric(df)
    categorical = get_cat(df)
    outliers = is_outlier(df)
    return analyzefe(
        missing_valuecount = missing,
        numeric_features = numeric,
        categorical_features = categorical,
        outliers_detection = outliers
    )


    

