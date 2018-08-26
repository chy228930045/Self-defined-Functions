# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 09:28:40 2018

@author: QY742WL
"""
import pandas as pd
import numpy as np

def trend_var_helper(x):
    trend_val = 0
    x = [item for item in x.tolist() if ~np.isnan(item)]
    for i in range(len(x)):
        if (i+1 - (1+len(x))*1.0/2) == 0:
            incremental_val = 0
        else:
            incremental_val = (-1)*(i+1 - (1+len(x))*1.0/2)*x[len(x)-1-i]/((i+1 - (1+len(x))*1.0/2)**2) 
        trend_val += incremental_val
    return trend_val 
    
def trend_var(df, id_col, dt_col, var_list=None, var_exclude=[]):
    if str(var_list) == "None":
        var_list = df.columns.tolist()
        
        for col in ([id_col] + var_exclude):
            var_list.remove(col)
    
    df=df.sort_values(by=[id_col, dt_col], ascending=[True, False])
    df_trend = df.groupby(id_col)[var_list].transform(trend_var_helper)
    df_trend = df_trend.reset_index()
    col_name = [i+'_trend' for i in var_list]
    df_trend.columns = [id_col] + col_name

    return df_trend

def moving_avg(df, id_col, dt_col, interval, var_list=None, var_exclude=[]): 
    if str(var_list) == "None":
        var_list = df.columns.tolist()
        
        for col in ([id_col, dt_col] + var_exclude):
            var_list.remove(col)
  
    df=df.sort_values(by=[id_col, dt_col], ascending=[True, False])
      
    rolling_df = df.groupby(id_col)[var_list].rolling(interval).mean()
    df_output = pd.DataFrame(df[id_col]).drop_duplicates()

    # min    
    df_temp = rolling_df.groupby(id_col)[var_list].min()
    col_name = [i+'_rolling_'+str(interval)+'m_min' for i in var_list]
    df_temp.columns=col_name
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")

    # max
    df_temp = rolling_df.groupby(id_col)[var_list].max()
    col_name = [i+'_rolling_'+str(interval)+'m_max' for i in var_list]
    df_temp.columns=col_name
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")

    # mean
    df_temp = rolling_df.groupby(id_col)[var_list].mean()
    col_name = [i+'_rolling_'+str(interval)+'m_mean' for i in var_list]
    df_temp.columns=col_name
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")
    
    # median
    df_temp = rolling_df.groupby(id_col)[var_list].median()
    col_name = [i+'_rolling_'+str(interval)+'m_median' for i in var_list]
    df_temp.columns=col_name
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")
    
    return df_output


def recent_avg(df, id_col, dt_col, period, var_list=None, var_exclude=[]): 
    
    if str(var_list) == "None":
        var_list = df.columns.tolist()
        
        for col in ([id_col, dt_col] + var_exclude):
            var_list.remove(col)
            
    df=df.sort_values(by=[id_col, dt_col], ascending=[True, False])
    
    recent_df = df.groupby(id_col, as_index=False)[[id_col]+var_list].head(period)
    df_output = pd.DataFrame(df[id_col]).drop_duplicates()
    
    # min    
    df_temp = recent_df.groupby(id_col)[var_list].min()
    col_name = [i+'_recent_'+str(period)+'m_min' for i in var_list]
    df_temp.columns=col_name
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")

    # max
    df_temp = recent_df.groupby(id_col)[var_list].max()
    col_name = [i+'_recent_'+str(period)+'m_max' for i in var_list]
    df_temp.columns=col_name
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")

    # mean
    df_temp = recent_df.groupby(id_col)[var_list].mean()
    col_name = [i+'_recent_'+str(period)+'m_mean' for i in var_list]
    df_temp.columns=col_name
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")
    
    return df_output


def mov_diff_avg(df, id_col, dt_col, period, interval, var_list=None, var_exclude=[]):
    
    if str(var_list) == "None":
        var_list = df.columns.tolist()
        
        for col in ([id_col, dt_col] + var_exclude):
            var_list.remove(col)
            
    df=df.sort_values(by=[id_col, dt_col], ascending=[True, False])
    
    mov_avg = pd.DataFrame(columns=[id_col, dt_col]+var_list)
    for i in range(period/interval):
        mov_avg_add = df.groupby(id_col, as_index=False).nth([i*interval+x for x in range(interval)]).groupby(id_col)[var_list].mean().reset_index()
        mov_avg = mov_avg.append(mov_avg_add)
   
    df_output = pd.DataFrame(df[id_col]).drop_duplicates()
    
    # min
    df_temp = mov_avg.groupby(id_col)[var_list].apply(lambda x: x.diff().min())
    df_temp.columns = [i+'_'+str(period)+'m_mov_diff_min' for i in var_list]
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")
    
    # max
    df_temp = mov_avg.groupby(id_col)[var_list].apply(lambda x: x.diff().max())
    df_temp.columns = [i+'_'+str(period)+'m_mov_diff_max' for i in var_list]
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")
    
    # mean
    df_temp = mov_avg.groupby(id_col)[var_list].apply(lambda x: x.diff().mean())
    df_temp.columns = [i+'_'+str(period)+'m_mov_diff_mean' for i in var_list]
    df_temp = df_temp.reset_index()
    df_output = df_output.merge(df_temp, on=id_col, how="left")
    
    return df_output
