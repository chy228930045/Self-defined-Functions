from collections import Counter
import pandas as pd

# Identify numeric and categorical variables
def EDD_type(df):
    numeric = list()
    categorical = list()
    
    for column in list(df): 
        try:
            df[column] = pd.to_numeric(df[column])
            numeric.append(column)
        except ValueError:
            categorical.append(column)
    
    return {'num':numeric, 'cat':categorical}

# Conduct EDD for numeric variables only
def EDD_num(df, ls_var=None, percentile=[.01, .05, .25, .5, .75, .95, .99]):
    
    # Calculate statistics
    if str(ls_var)=='None': 
        output_df = df[EDD_type(df).get('num')].describe(percentile).transpose()
    else: 
        output_df = df[ls_var].describe(percentile).transpose()
    
    output_df = output_df.rename(index=str, columns={"count": "NonmissingCt"})

    output_df.insert(0, 'DataType', 'Numeric')
    output_df.insert(1, 'Nobs', df.shape[0])
    output_df.insert(2, 'MissingCt', output_df['Nobs']-output_df['NonmissingCt'])
    
    MissingPctg = output_df['MissingCt']*1.0/output_df['Nobs']
    output_df.insert(3, 'MissingPctg', ['{0:.2%}'.format(i) for i in MissingPctg])
    
    UniqueCt = df.T.apply(lambda x: x.nunique(), axis=1)
    output_df.insert(4, 'UniqueCt', UniqueCt)
    output_df = output_df.reset_index()
    output_df = output_df.rename(index=str, columns={"index": "Feature"})
    
    return output_df

# Conduct EDD for ls_varal variables only
def EDD_cat(df, ls_var=None, top_count=5): 
    if top_count > 20: 
        raise ValueError('Maximum value for top_count is 20!')
        
    if str(ls_var)=='None': 
        ls_var = EDD_type(df).get('cat')
    
    col=['Feature', 'Type', 'Nobs', 'UniqueCt', 'NonmissingCt', 'MissingCt', 'MissingPctg']
    for i in range(top_count):
        col.append('Top'+str(i+1))
        col.append('Top'+str(i+1)+'_Freq')
    output_df = pd.DataFrame(columns=col)

    for ct in range(len(ls_var)):
        f = ls_var[ct]
        lst = df[f][df[f].notnull()]
        vals = Counter(lst).most_common(top_count)
        op_cm = [f, 'Categorical', df.shape[0], df[f].nunique(), len(lst), df.shape[0]-len(lst), '{0:.2%}'.format(1-float(len(lst))/df.shape[0])]
        if vals[0][1] == 1:
            output_df.loc[ct] = op_cm+['Primary Key' for i in range(top_count*2)]
        else:
            flatten_vals = [e for l in vals for e in l]
            output_df.loc[ct] = op_cm + flatten_vals + [None for i in range(top_count*2-len(flatten_vals))]
    
    return output_df


"""
    Usage: input A) a Pandas dataframe; 
                 B) a list of columns to be include in the analysis   
                 C) which variables to force to categoric, if any - Optional; 
                 D) percentile thresholds for numeric variables - Optional; 
                 E) number of top frequency items for categoric variables - Optional.
    Output: 2 dataframes (numeric EDD and categorical EDD) 
    Example:
        EDD(svcgfile_Q12001)
"""

def EDD(df, ls_var=None, ls_force_categorical=None, percentile=[.01, .05, .25, .5, .75, .95, .99], top_count=5): 
    if str(ls_var)=='None': 
        ls_var = df.columns.tolist()    
    input_types = EDD_type(df[ls_var])
    
    if str(ls_force_categorical) <> 'None': 
        if type(ls_force_categorical) is not list:
            input_types.get('num').remove(ls_force_categorical)
            input_types.get('cat').append(ls_force_categorical)
        else: 
            for i in ls_force_categorical: 
                input_types.get('num').remove(i)
                input_types.get('cat').append(i)
    
    if len(input_types.get('num')) <> 0:
        df_numeric = EDD_num(df, input_types.get('num'), percentile)
    else:
        df_numeric = None
        
    if len(input_types.get('cat')) <> 0:
        df_categorical = EDD_cat(df, input_types.get('cat'), top_count)
    else:
        df_categorical = None
    
    return df_numeric, df_categorical