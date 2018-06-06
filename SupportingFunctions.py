import cPickle
import time

def save_model(model, path, file_name):
	'''
	Date		:	04/19/2018 
	
	Description	:	Save machine learning model into a file
	
	Parameters	:	model - Sklearn model object
					path - string	{r"\folder\subfolder"}
					file_name - string
	
	Return		:	A binary file that stores all model parameters
	
	Example		:	save_model(model_xg, r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Deliverable\model", "lal_model_xgboost")
	'''

	start = time.time()

	path = path.replace('\\', '/')
	path = path + "/" + file_name
	with open(path, 'wb') as f:
		cPickle.dump(model, f)
	
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')

def load_model(path):
	'''
	Date		:	04/19/2018 
	
	Description	:	Load machine learning model into python
	
	Parameters	:	path - string	{r"\folder\subfolder\model_file"}
	
	Return		:	A model object
	
	Example		:	load_model(r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Deliverable\model\lal_model_xgboost")
	'''
	
	start = time.time()
	
	path = path.replace('\\', '/')
	with open(path,'rb') as model_file:
		model = cPickle.load(model_file)
		
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')
	
	return model
	
#######################################################################################
import pandas as pd
def load_csv(path):
	'''
	Date		:	04/19/2018 
	
	Description	:	Load csv into pandas dataframe
	
	Parameters	:	path - string	{r"\folder\subfolder\csvfile.csv"}
	
	Return		:	A dataframe
	
	Example		:	load_model(r"\\v\region\na\appl\banking\pbg_analytics_prql\data\prod_non_pii\Hongyu\Deliverable\model\abc.csv")
	'''
	
	start = time.time()
	
	path = path.replace('\\', '/')
	with open(path) as fp:
		# define missing value
		df = pd.read_csv(fp, na_values=['', '.', 'NaN', 'NA'])
	
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')
	print('dataframe dimension: ' + str(df.shape))
	return df

import sklearn
from sklearn.metrics import roc_curve, auc
def cal_roc(model, x, y):
    '''
	Date		:	04/19/2018 
	
	Description	:	Calculate ROC
	
	Parameters	:	model - Sklearn model object
					x - Dataframe
						Dataset with only independent variables
					y - Dataframe 
						Dataset with only the dependent variable
					
	Return		:	float {AUC}
					List {fpr}
					List {tpr}
	
    Example		:	N/A
    '''    
    probs_true = model.predict_proba(x)[::, 1]
    fpr, tpr, thresholds = roc_curve(y, probs_true)
    roc_auc = auc(fpr,tpr)
    	
    return roc_auc, fpr, tpr

import matplotlib
import matplotlib.pyplot as plt
def plot_roc(model, x, x_val, y, y_val, path_save=None, file_name=None):
	'''
	Date		:	04/19/2018 
	
	Description	:	Plot ROC curve for both training and testing datasets
	
	Parameters	:	model - Sklearn model object
					x - Dataframe
						Training dataset with only independent variables
					x_val - Dataframe
							Testing dataset with only independent variables
					y - Dataframe 
						Training dataset with only the dependent variable
					y_val - Dataframe
							Testing dataset with only the dependent variable
					path_save - String (optional) {r"\folder\subfolder"}
								The location for saving the chart
					file_name - String {"filename"}
					
	Return		:	N/A
	
	Example		:	N/A
	'''
	
	roc_auc, fpr, tpr = cal_roc(model, x, y)
	roc_auc_val, fpr_val, tpr_val = cal_roc(model, x_val, y_val)
	
	plt.clf()
	plt.plot(fpr, tpr, label='ROC Curve Train (area=%0.3f)' % roc_auc)
	plt.plot(fpr_val, tpr_val, label='ROC Curve Validation (area=%0.3f)' % roc_auc_val)
	
	# add a diagonal line
	plt.plot([0,1],[0,1], 'k--')
	plt.xlim (-0.05, 1.05)
	plt.ylim (-0.05, 1.05)
	plt.xlabel('False Positve Rate')
	plt.ylabel('True Positve Rate')
	plt.title("ROC - " + model.__class__.__name__)
	
	# set the legend location
	plt.legend(loc="lower right")
	if (path_save != None) & (file_name !=None):
		plt.savefig(path_save.replace("\\", "/") + "/" + file_name + ".png")
	plt.show()

import numpy as np
def cal_ks_lift(model, x, y, num_bins):
	'''
	Date		:	04/19/2018 
	
	Description	:	Calculate KS-statistics and prepare lift table
	
	Parameters	:	model - Sklearn model object
					x - Dataframe
						Dataset with only independent variables
					y - Dataframe 
						Dataset with only the dependent variable
					num_bins - Integer (optional)
					
	Return		:	float {KS value}
					dataframe {lift table}
	
	Example		:	N/A
	'''
		
	probs = model.predict_proba(x)
	probs_true = probs[:, 1]
	
	df_y_prob = pd.DataFrame(probs_true, columns = ['y_prob'])
	df_y = pd.DataFrame(y.values, columns = ['y'])
	df_lift = pd.concat([df_y_prob, df_y], 1)
	
	bins = np.unique(df_lift['y_prob'].quantile(np.linspace(0, 1, num_bins)))
	df_lift['decile'] = pd.cut(df_lift['y_prob'], bins, labels=np.arange(0,len(bins)-1,1)[::-1], include_lowest=True)
	df_lift = df_lift.groupby(['decile','y'])
	df_lift = df_lift.y.count().unstack().fillna(0).sort_index(ascending=False)
	df_lift.columns = ['good', 'bad']
	total_good, total_bad = df_lift.sum()[0], df_lift.sum()[1]
	
	df_lift['dist_good'] = df_lift['good'] / total_good
	df_lift['dist_bad'] = df_lift['bad'] / total_bad
	df_lift['dist_total'] = (df_lift['good'] + df_lift['bad']) / (total_good + total_bad)
	
	df_lift['cul_dist_good'] = df_lift['dist_good'].cumsum(axis=0)
	df_lift['cul_dist_bad'] = df_lift['dist_bad'].cumsum(axis=0)
	df_lift['cul_dist_total'] = df_lift['dist_total'].cumsum(axis=0)
	
	df_lift['cul_lift'] = (df_lift['cul_dist_bad'] + 0.0001)/(df_lift['cul_dist_total'] + 0.0001)
	
	KS_value = max(abs(df_lift['cul_dist_good'] - df_lift['cul_dist_bad']))
	
	print(model.__class__.__name__) + ":"
	print ('KS value:{0}'.format(KS_value))
	
	return KS_value, df_lift

def plot_gain_lift(model, x, x_val, y, y_val, num_bins=21):
	'''
	Date		:	04/19/2018 
	
	Description	:	Plot gain chart and lift chart for both training and testing datasets
	
	Parameters	:	model - Sklearn model object
					x - Dataframe
						Training dataset with only independent variables
					x_val - Dataframe
						Testing dataset with only independent variables
					y - Dataframe 
						Training dataset with only the dependent variable
					x_val - Dataframe
							Testing dataset with only the dependent variable
					num_bins - Integer (optional)
					
	Return		:	N/A
	
	Example		:	N/A
	'''
	
	print("Training:")
	KS_value, df_lift = cal_ks_lift(model, x, y, num_bins)
	
	print("\nValidation:")
	KS_value_val, df_lift_val = cal_ks_lift(model, x_val, y_val, num_bins)
	
	# Gain Chart
	plt.clf()
	plt.plot([0] + df_lift['cul_dist_total'].tolist(), [0] + df_lift['cul_dist_bad'].tolist(), label='Training')
	plt.plot([0] + df_lift_val['cul_dist_total'].tolist(), [0] + df_lift_val['cul_dist_bad'].tolist(), label='Validation')
	
	plt.plot([0,1],[0,1], 'k--')
	plt.xlim (-0.05, 1.05)
	plt.ylim (-0.05, 1.05)
	
	plt.xlabel('% Population')
	plt.ylabel('% Expected Responses')
	plt.title('Cumulative Gains Chart')
	plt.legend(loc="lower right")
	plt.show()
	
	# Lift Chart
	plt.clf()
	plt.plot(df_lift['cul_dist_total'], df_lift['cul_lift'], label='Training')
	plt.plot(df_lift_val['cul_dist_total'], df_lift_val['cul_lift'], label='Validation')
	
	plt.hlines(y=1, xmin=0, xmax=1, color='k', linestyle='--')
	plt.xlim (-0.05, 1.05)
	
	plt.xlabel('% Population')
	plt.ylabel('Lift')
	plt.title('Lift Chart')
	plt.legend(loc="upper right")
	plt.show()
	
def model_performance_01(model, x, y):
	df_prob = pd.DataFrame()
	df_prob['y'] = y.values
	df_prob['prob'] = model.predict_proba(x)[::, 1]
	df_prob['ind'] = 1
	
	df_prob = df_prob.sort('prob', ascending=False)
	df_prob['obs_rrate'] = df_prob.y.sum() * 1.0 / df_prob.y.count()
	df_prob['cum_obs_rate'] = df_prob.y.cumsum() * 1.0 / df_prob.ind.cumsum()
	df_prob['cum_obs_rate'] = np.where(df_prob.y.cumsum()==0,df_prob['cum_obs_rate'].max(),df_prob['cum_obs_rate'])
	df_prob['lift'] = df_prob.cum_obs_rate / df_prob.obs_rrate
	
	print('Total', len(df_prob))
	print('3xtimes', len(df_prob[df_prob.lift>=3]))
	print('4xtimes', len(df_prob[df_prob.lift>=4]))
	print('5xtimes', len(df_prob[df_prob.lift>=5]))

from pandas.core.algorithms import algos
def partial_dependency_1d(model, x, feature = None, num_bins = 10, path_save=None):
	'''
	Date		:	05/03/2018 
	
	Description	:	Plot 1-d partial dependency chart
	
	Parameters	:	model - Sklearn model object
					x - Dataframe
						validation dataset with only independent variables
					feature - List
						A list of variables the user is interested in. If None, PDP will be created for all variables 
					num_bins - Integer 
						# of dots for the plot
					path_save - String (optional) {r"\folder\subfolder"}
								The location for saving the chart
					
	Return		:	N/A
	
	Example		:	N/A
	'''
		
	y_pred = model.predict_proba(x)[::, 1]
	
	if feature == None:
		feature = x.columns.values.tolist()
		print('No feature is specificed. All the available features are used\n')
	
	for var in feature:
		x_temp = x.copy()
		df_var = x_temp[[var]].dropna()
		grid = np.unique(algos.quantile(df_var, np.linspace(0.01, 0.99, num_bins)))
		y_pred_temp = np.zeros(len(grid))
		
		for i, value in enumerate(grid):
			x_temp[var] = value
			probs = model.predict_proba(x_temp)
			y_pred_temp[i] = np.average(probs[::,1])
			
		# draw the pdp
		fig, ax = plt.subplots()
		fig.set_size_inches(7, 7)
		plt.subplots_adjust(left = 0.17, right = 0.94, bottom = 0.15, top = 0.9)

		ax.plot(grid, y_pred_temp, '-', color = 'red', linewidth = 2.5, label='avg.', marker='o')
		ax.plot(x[var], y_pred, 'o', color = 'grey', alpha = 0.01)
			
		# Adjust axes
		ax.set_xlim(min(grid)-(max(grid)-min(grid))/50, max(grid)+(max(grid)-min(grid))/50)
		ax.set_ylim(min(y_pred_temp)-(max(y_pred_temp)-min(y_pred_temp))/50, max(y_pred_temp)+(max(y_pred_temp)-min(y_pred_temp))/50)
		ax.set_xlabel(var, fontsize = 10)
		ax.set_ylabel('Partial Dependence Plot - Prob.', fontsize = 12)
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels, loc = 'best', fontsize = 12)

		# Save the graph
		if path_save != None:
			plt.savefig(path_save.replace("\\", "/") + "/" + var + ".png")

		plt.show()

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata

def partial_dependency_2d(model, x, feature, num_bins = 10, path_save=None):
	'''
	Date		:	05/03/2018 

	Description	:	Plot 2-d partial dependency chart

	Parameters	:	model - Sklearn model object
					x - Dataframe
						validation dataset with only independent variables
					feature - Dataframe
						A dataframe with pairs of variables and relative rank
						e.g. var1, var2, rank_1, rank_2
						The following functions can be used to get pairs.
						
						def get_pairs(df, var):
							df = df.reset_index()
							pairs = set()
							for i in range(df.shape[0] - 1):
								for j in range(i+1, df.shape[0]):
									pairs.add((df.loc[i,var], df.loc[j,var], i+1, j+1))
							pairs = pd.DataFrame(list(pairs))
							pairs.columns = ["var_1", "var_2", "rank_1", "rank_2"]
							return pairs

						pairs_2 = get_pairs(imp_rf, "feature")
						pairs_2 = pairs_2[(pairs_2.rank_1 <= 10) & (pairs_2.rank_2 <= 10)]

					num_bins - Integer 
						# of dots for the plot
					path_save - String (optional) {r"\folder\subfolder"}
								The location for saving the chart
					
	Return		:	N/A

	Example		:	N/A
	'''
		
	for i in range(feature.shape[0]):
		x_temp = x.copy()
		feature_1 = feature['var_1'].iloc[i]
		feature_1_rank = feature['rank_1'].iloc[i]
		
		feature_2 = feature['var_2'].iloc[i]
		feature_2_rank = feature['rank_2'].iloc[i]
		
		df_var_1 = x_temp[[feature_1]].dropna()
		df_var_2 = x_temp[[feature_2]].dropna()

		grid_1 = np.unique(algos.quantile(df_var_1, np.linspace(0.01, 0.99, num_bins)))
		grid_2 = np.unique(algos.quantile(df_var_2, np.linspace(0.01, 0.99, num_bins)))

		y_pred_temp = np.zeros(len(grid_1)*len(grid_2))
		grid_1_temp = np.zeros(len(grid_1)*len(grid_2))
		grid_2_temp = np.zeros(len(grid_1)*len(grid_2))
		
		j = 0 
		for i_1, value_1 in enumerate(grid_1):
			for i_2, value_2 in enumerate(grid_2):               
				x_temp[feature_1] = value_1
				x_temp[feature_2] = value_2
				
				probs=model.predict_proba(x_temp)
				y_pred_temp[j] = np.average(probs[::,1])
				grid_1_temp[j] = value_1
				grid_2_temp[j] = value_2
				
				j += 1

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		fig.set_size_inches(17, 10)
		ax.view_init(20,10)

		# 2D-arrays from DataFrame
		x_feature_1 = np.linspace(grid_1_temp.min(), grid_1_temp.max(), len(grid_1_temp))
		x_feature_2 = np.linspace(grid_2_temp.min(), grid_2_temp.max(), len(grid_2_temp))
		x1, x2 = np.meshgrid(x_feature_1, x_feature_2)

		# Interpolate unstructured D-dimensional data.
		z = griddata((grid_1_temp, grid_2_temp), y_pred_temp, (x1, x2), rescale=False)

		# Plot the surface.
		surf = ax.plot_surface(x1, x2, z, rstride=1, cstride=5, cmap=cm.coolwarm,shade=True,
							   linewidth=0, antialiased=True)

		# Customize the z axis.
		ax.set_zlim(np.min(z)-0.005, np.max(z)+0.005)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

		# Add a color bar which maps values to colors.
		fig.colorbar(surf, shrink=0.5, aspect=5)

		ax.set_xlabel(feature_1,fontsize=10)
		ax.set_ylabel(feature_2, fontsize=10)
		ax.set_zlabel('Prob.', fontsize=10)

		# Save the graph
		file_name =  str(feature_1_rank) + "&" +  str(feature_2_rank) + " - " + feature_1 + "&" + feature_2
		if path_save != None:
			plt.savefig(path_save.replace("\\", "/") + "/" + file_name + ".png")
			
		plt.show()