import pandas as pd
import numpy as np
from io import StringIO
from dataclasses import dataclass



def describe(df):
    return df.describe().T
def cal_mean(df):
    return df.mean(numeric_only = True)
def cal_median(df):
    return df.median(numeric_only = True)
def cal_mode(df):
    return df.mode().iloc[0]
def variance(df):
    return df.var(numeric_only = True)
def standard_deviation(df):
    return df.std(numeric_only = True)
def corr_matrix(df, method = "pearson"):
   """ method:
        pearson
        spearman
        kendall
        """
   return df.corr(numeric_only = True, method = method )
def covarience_matrix(df):
    return df.cov(numeric_only = True)
def skewnwss(df):
    return df.skew(numeric_only = True)
def kurtosis(df):
    return df.kurt(numeric_only = True)
def unique_values(df):
    return df.nunique()
def data_types(df):
    return df.dtypes
def dataset_shape(df):
    return df.shape
def dataset_info(df):
    buffer = StringIO()
    df.info(buf = buffer)
    return buffer.getvalue()
@dataclass
class StatisticsReport:   
    shape: tuple
    summary: pd.DataFrame
    mode: pd.DataFrame
    unique_values: pd.DataFrame
    describe: pd.DataFrame
    correlation: pd.DataFrame
    info: str
    
    

def statistics_summary(df):
    unique=unique_values(df),
    describ=describe(df)
    shape = dataset_shape(df)
    information = dataset_info(df)
    correlation=corr_matrix(df)
    mode=cal_mode(df)
    
    summary = pd.DataFrame({
     
        "mean":cal_mean(df),
        "median":cal_median(df),
        "varian":variance(df),
        "standard_deviation":standard_deviation(df),
        "skew":skewnwss(df),
        "kurtos":kurtosis(df),
    })
    return StatisticsReport(
        shape = shape,
        summary = summary,
        mode = mode,
        unique_values= unique,
        describe = describ,
        correlation = correlation,
        info = information,  
    )

 



