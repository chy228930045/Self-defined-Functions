# data loading - ALL
Master_DS_250K_600K = load_csv(r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Python\SBL-LAL Model\Master_DS_v1_samp_250k.csv")
Master_DS_250K_600K = Master_DS_250K_600K[Master_DS_250K_600K.has_lal==0]
print "dataframe dimension after filtering" + str(Master_DS_250K_600K.shape)

pd.set_option("display.max_columns",10)
Master_DS_250K_600K.head(10)
###############################################################################################################

# split datasets into training and testing
x, x_val, y, y_val = train_test_split(Model_DS_250K_600K[var_x_internal],
                                      Model_DS_250K_600K[var_y], test_size=0.33, random_state=42)

print('Training/Testing Set Dimension - X:')
print('Training: ' + str(x.shape))
print('Testing: ' + str(x_val.shape) + '\n')

print('Training/Testing Set Dimension - Y:')
print('Training: ' + str(y.shape))
print('Testing: ' + str(y_val.shape))
###############################################################################################################

# RF variable selection
model_rf_temp, imp_rf_temp = my_sklearn_rf(x, y, param = {"n_estimators":100, 
                                                          "max_features":"auto", 
                                                          "max_depth":500, 
                                                          "min_samples_split":0.001,
                                                          "min_samples_leaf":0.0001,
                                                          "random_state":666})
plot_roc(model_rf_temp, x, x_val, y, y_val, path_save=None, file_name=None)
plot_gain_lift(model_rf_temp, x, x_val, y, y_val, num_bins=21)

dir_out = r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Data\LAL Model\250K 600K"
imp_rf_temp['importance_pctg'] = imp_rf_temp['importance'] / imp_rf_temp['importance'].max()
filename = dir_out + "\\rf_importance_internal.csv"
filename = filename.replace('\\', '/')
imp_rf_temp.to_csv(filename, index=False)

###############################################################################################################

# Remove low importance variables
print '# of variables before RF selection: %s' % len(var_x_internal)
var_drop_rf = imp_rf_temp.loc[imp_rf_temp.importance_pctg < 0.2, 'feature']
for item in var_drop_rf:
    try:
        var_x_internal.remove(item)
    except:
        continue
print '# of variables after RF selection: %s' % len(var_x_internal)

###############################################################################################################

# Variable reduction - IV
df = pd.concat([x[var_x_internal], y], axis=1)
df_woe = cal_woe_iv(df = df, var_x = var_x_internal, var_y = ["new_lal_target_t3_t8"])
df_iv_internal = df_woe[["variable_name", "IV"]].drop_duplicates()
df_iv_internal.head(10)

# Correlation analysis
corr_list = my_corr(x[var_x_internal], "pearson", abs=True)
corr_list = corr_list.reset_index()
corr_list.columns = ['var_1', 'var_2', 'corr']

# Add IV
corr_list = pd.merge(corr_list, df_iv_internal, how='left', left_on='var_1', right_on='variable_name')
corr_list = corr_list.drop(['variable_name'], axis=1)

corr_list = corr_list.rename(columns={'IV':'var_1_iv'})
corr_list = pd.merge(corr_list, df_iv_internal, how='left', left_on='var_2', right_on='variable_name')
corr_list = corr_list.drop(['variable_name'], axis=1)
corr_list = corr_list.rename(columns={'IV':'var_2_iv'})
corr_list = corr_list.sort('corr', ascending=False)

filename = dir_out + "\\df_corr_analysis.csv"
filename = filename.replace('\\', '/')
corr_list.to_csv(filename, index=False)
print corr_list.head(20)

###############################################################################################################

## Remove highly correlated and low IV variables
print '# of variables before Corr selection: %s' % len(var_x_internal)
corr_list.loc[corr_list.var_1_iv >= corr_list.var_2_iv, 'ToDrop'] = corr_list.var_2
corr_list.loc[corr_list.var_1_iv < corr_list.var_2_iv, 'ToDrop'] = corr_list.var_1
cutoff = 0.6
var_drop_corr = list(set(corr_list.loc[(corr_list["corr"] >= cutoff), "ToDrop"]))
for item in var_drop_corr:
    var_x_internal.remove(item)   
print '# of variables after Corr selection: %s' % len(var_x_internal)

# Remove low IV variables
print '# of variables before IV selection: %s' % len(var_x_internal)
var_drop_iv = df_iv_internal.loc[df_iv_internal.IV < 0.1, 'variable_name']
for item in var_drop_iv:
    try:
        var_x_internal.remove(item)
    except:
        continue
print '# of variables after IV selection: %s' % len(var_x_internal)
###############################################################################################################

# Export final variables
df_iv_internal_final = df_iv_internal[df_iv_internal["variable_name"].isin(var_x_internal)]
df_iv_internal_final.sort('IV', ascending = False)
filename = dir_out + "\\final_variable.csv"
filename = filename.replace('\\', '/')
df_iv_internal_final.to_csv(filename, index=False)

final_woe = df_woe[df_woe["variable_name"].isin(var_x_internal)]
filename = dir_out + "\\final_variable_woe.csv"
filename = filename.replace('\\', '/')
final_woe.to_csv(filename, index=False)

x = x[var_x_internal]
x_val = x_val[var_x_internal]

###############################################################################################################
# Modelling - RF
model_rf, imp_rf = my_sklearn_rf(x, y, param = {"n_estimators":500, 
                                                "max_features":8, 
                                                "max_depth":30, 
                                                "min_samples_split":0.001,
                                                "min_samples_leaf":0.00005,
                                                "random_state":666})
plot_roc(model_rf, x, x_val, y, y_val, path_save=None, file_name=None)
plot_gain_lift(model_rf, x, x_val, y, y_val, num_bins=21)

filename = dir_out + "\\final_variable_RF.csv"
filename = filename.replace('\\', '/')
imp_rf.to_csv(filename, index=False)

save_model(model_rf, r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Deliverable\model\250K 600K", "lal_model_rf")

# Modelling - Logistic
model_log, imp_log = my_sklearn_logistic(x, y, param={"penalty":"l1", "C":50.0, "random_state":666})
plot_roc(model_log, x, x_val, y, y_val, path_save=None, file_name=None)
plot_gain_lift(model_log, x, x_val, y, y_val, num_bins=21)

filename = dir_out + "\\final_variable_Log.csv"
filename = filename.replace('\\', '/')
imp_log.to_csv(filename, index=False)

save_model(model_log, r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Deliverable\model\250K 600K", "lal_model_log")

# Modelling - Tree
model_dtree, imp_dtree = my_sklearn_dtree(x, y, param = {"max_features":8, 
                                                         "max_depth":7, 
                                                         "min_samples_split":0.001,
                                                         "min_samples_leaf":0.0001,
                                                         "random_state":666})
plot_roc(model_dtree, x, x_val, y, y_val, path_save=None, file_name=None)
plot_gain_lift(model_dtree, x, x_val, y, y_val, num_bins=21)

filename = dir_out + "\\final_variable_dtree.csv"
filename = filename.replace('\\', '/')
imp_dtree.to_csv(filename, index=False)

save_model(model_dtree, r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Deliverable\model\250K 600K", "lal_model_dtree")

# Modelling - XGboost
model_xgb, imp_xgb = my_API_XGBoost(x, y, x_val, y_val, param={"learning_rate":0.08,
											  "n_estimators":100, 
											  "max_depth":8, 
											  "colsample_bytree":0.95,
											  "colsample_bylevel":0.6,
											  "random_state":666})
plot_roc(model_xgb, x, x_val, y, y_val, path_save=None, file_name=None)
plot_gain_lift(model_xgb, x, x_val, y, y_val, num_bins=21)

filename = dir_out + "\\final_variable_xgb.csv"
filename = filename.replace('\\', '/')
imp_xgb.to_csv(filename, index=False)

save_model(model_xgb, r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Deliverable\model\250K 600K", "lal_model_xgb")

###############################################################################################################

# ROC Comparison
def cal_roc(model, x, y):
    probs_true = model.predict_proba(x)[::,1]
    fpr, tpr, thresholds = roc_curve(y, probs_true)
    roc_auc = auc(fpr,tpr)
    return roc_auc, fpr, tpr

get_ipython().magic(u'matplotlib inline')
plt.clf()
plt.figure(figsize=(20,15))

roc_auc, fpr, tpr = cal_roc(model_rf, x_val, y_val)
plt.plot(fpr, tpr, label='ROC Curve RF - Val Sample (area=%0.3f)' % roc_auc)

roc_auc, fpr, tpr = cal_roc(model_xgb, x_val, y_val)
plt.plot(fpr, tpr, label='ROC Curve XGB - Val Sample (area=%0.3f)' % roc_auc)

roc_auc, fpr, tpr = cal_roc(model_log, x, y)
plt.plot(fpr, tpr, label='ROC Curve Logistic - Train Sample (area=%0.3f)' % roc_auc)

roc_auc, fpr, tpr = cal_roc(model_log, x_val, y_val)
plt.plot(fpr, tpr, label='ROC Curve Logistic - Val Sample (area=%0.3f)' % roc_auc)

roc_auc, fpr, tpr = cal_roc(model_dtree, x, y)
plt.plot(fpr, tpr, label='ROC Curve Tree - Train Sample (area=%0.3f)' % roc_auc)

roc_auc, fpr, tpr = cal_roc(model_dtree, x_val, y_val)
plt.plot(fpr, tpr, label='ROC Curve Tree - Val Sample (area=%0.3f)' % roc_auc)


plt.plot([0,1],[0,1], 'k--')
plt.xlim (0.0, 1.05)
plt.ylim (0.0, 1.05)
plt.xlabel('False Positve Rate')
plt.ylabel('True Positve Rate')
plt.title("Model AUC Comparison")
plt.legend(loc="lower right")
plt.show()

# Lift Comparison
get_ipython().magic(u'matplotlib inline')
plt.clf()
plt.figure(figsize=(20,15))

print "RF Val"
KS_value, df_lift = cal_ks_lift(model_rf, x_val, y_val)
plt.plot(df_lift['cul_dist_total'], df_lift['cul_lift'], label='RF - Val Sample')
print df_lift
print "="*100

print "XGB Val"
KS_value, df_lift = cal_ks_lift(model_xgb, x_val, y_val)
plt.plot(df_lift['cul_dist_total'], df_lift['cul_lift'], label='XGB - Val Sample')
print df_lift
print "="*100

print "Log Train"
KS_value, df_lift = cal_ks_lift(model_log, x, y)
plt.plot(df_lift['cul_dist_total'], df_lift['cul_lift'], label='Logistic - Training Sample')
print df_lift
print "="*100

print "Log Val"
KS_value, df_lift = cal_ks_lift(model_log, x_val, y_val)
plt.plot(df_lift['cul_dist_total'], df_lift['cul_lift'], label='Logistic - Val Sample')
print df_lift
print "="*100

print "Tree Train"
KS_value, df_lift = cal_ks_lift(model_dtree, x, y)
plt.plot(df_lift['cul_dist_total'], df_lift['cul_lift'], label='Tree - Training Sample')
print df_lift
print "="*100

print "Tree Val"
KS_value, df_lift = cal_ks_lift(model_dtree, x_val, y_val)
plt.plot(df_lift['cul_dist_total'], df_lift['cul_lift'], label='Tree - Val Sample')
print df_lift
print "="*100

plt.xlim([0.0, 1.05])
plt.xlabel('% Population')
plt.ylabel('Lift')
plt.title('Lift Curve')
plt.legend(loc="upper right")
plt.show()

# num of leads from model
print "RF - Val"   
model_performance_01(model_rf, x_val, y_val)

print "\nXGB - Val"   
model_performance_01(model_xgb, x_val, y_val)

print "\nLog - Train"   
model_performance_01(model_log, x, y)

print "\nLog - Val"   
model_performance_01(model_log, x_val, y_val)

print "\nTree - Train"   
model_performance_01(model_dtree, x, y)

print "\nTree - Val"   
model_performance_01(model_dtree, x_val, y_val)