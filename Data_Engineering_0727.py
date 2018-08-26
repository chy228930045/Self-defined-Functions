# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:28:40 2018

@author: QY742WL
"""

import numpy as np

def trend_var(x):
    # x must be sorted by date
    trend_val = 0
    x = [item for item in x.tolist() if ~np.isnan(item)]
    for i in range(len(x)):
        if (i+1 - (1+len(x))*1.0/2) == 0:
            incremental_val = 0
        else:
            incremental_val = (-1)*(i+1 - (1+len(x))*1.0/2)*x[len(x)-1-i]/((i+1 - (1+len(x))*1.0/2)**2) 
        trend_val += incremental_val
    return trend_val 


def derive_min(df, var_lst, id_col, dt_col, mths): 
    
    df=df.sort_values(by=[id_col, dt_col])
    
    op_col = [i+'_'+str(mths)+'_month_min' for i in var_lst]
    
    rolling_df = df[var_lst].rolling(window=mths).min()
    rolling_df.columns=op_col
    df=df.join(rolling_df)
    df.loc[df.groupby(id_col).head(2).index.values, op_col]=np.nan
    
    return df

    
def derive_max(df, var_lst, id_col, dt_col, mths): 
    
    df=df.sort_values(by=[id_col, dt_col])
    
    op_col = [i+'_'+str(mths)+'_month_max' for i in var_lst]
    
    rolling_df = df[var_lst].rolling(window=mths).max()
    rolling_df.columns=op_col
    df=df.join(rolling_df)
    df.loc[df.groupby(id_col).head(2).index.values, op_col]=np.nan
    
    return df


def derive_avg(df, var_lst, id_col, dt_col, mths): 
    
    df=df.sort_values(by=[id_col, dt_col])
    
    op_col = [i+'_'+str(mths)+'_month_avg' for i in var_lst]
    
    rolling_df = df[var_lst].rolling(window=mths).mean()
    rolling_df.columns=op_col
    df=df.join(rolling_df)
    df.loc[df.groupby(id_col).head(2).index.values, op_col]=np.nan
    
    return df
    