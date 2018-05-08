def my_sklearn_rf(x, y, param = {"n_estimators":100, 
								 "max_features":"auto", 
								 "max_depth":10, 
								 "min_samples_split":0.001,
								 "min_samples_leaf":0.0001,
								 "random_state":66}):
	
	'''
	Date		:	04/20/2018 
	
	Description	:	Apply random forest model
	
	Parameters	:	x - Dataframe
						Training dataset with only independent variables
					y - Dataframe 
						Training dataset with only the dependent variable
					param - Dictionary (optional)
							Parameters used for the model. Check the link below for details.
							http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
					
	Return		:	model object
					dataframe
	
	Example		:	N/A
	'''
	
	start = time.time()
	print("#### Parameters for RF ####")
	print("n estimators: %d" % param["n_estimators"])
	print("max features: %s" % param["max_features"])
	print("max depth: %d" % param["max_depth"])
	print("min sample split: %0.05f" % param["min_samples_split"])
	print("min sample leaf: %0.05f" % param["min_samples_leaf"])
	print("###########################\n")
	
	# apply random forest model
	rf_model = rf(n_estimators=param["n_estimators"],
				max_features=param["max_features"],
				max_depth=param["max_depth"], 
				min_samples_split=param["min_samples_split"], 
				min_samples_leaf=param["min_samples_leaf"],
				random_state = 666,
				n_jobs=48)
	
	imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
	steps =[('imputation', imp),('RandomForestClassifier', rf_model)]
	model = Pipeline(steps)
	model.fit(x,y)
	
	# feature importance
	df_importance = pd.DataFrame({'feature':x.columns, 'importance':np.round(model.named_steps['RandomForestClassifier'].feature_importances_, 6)})
	df_importance = df_importance.sort('importance',ascending=False)
	df_importance = df_importance.reset_index()
	df_importance = df_importance.drop(['index'], axis=1)
	
	# plot top 20 important features
	df_importance[df_importance.index < 20].plot(kind='barh', x='feature', y='importance', legend=False, figsize=(6, 10))
	plt.xlabel('importance value')
	plt.ylabel('feature')
	plt.gca().invert_yaxis()
	plt.show()
	
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')
	
	return model, df_importance
	
def my_sklearn_logistic(x, y, param={"penalty":"l1",
									 "C":50.0,
									 "random_state":66}):
	
	'''
	Date		:	04/20/2018 
	
	Description	:	Apply logistic model
	
	Parameters	:	x - Dataframe
						Training dataset with only independent variables
					y - Dataframe 
						Training dataset with only the dependent variable
					param - Dictionary (optional)
							Parameters used for the model. Check the link below for details.
							http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
					
	Return		:	model object
					dataframe
	
	Example		:	N/A
	'''
	
	start = time.time()
	print("#### Parameters for Logistic ####")
	print("penalty %s" % param["penalty"])
	print("C: %0.05f" % param["C"])
	print("###########################\n")
	
	start = time.time()
	
	imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
	logistic = linear_model.LogisticRegression(penalty=param["penalty"], C=param["C"], random_state=param["random_state"], n_jobs=50)
	model = Pipeline([('imputation', imp),('LogisticRegression', logistic)])
	model.fit(x, y)
	
	#Calculate pValue
	coef = np.append(logistic.intercept_, logistic.coef_)
	predictions = model.predict(x)
	newX = pd.DataFrame({"Constant":np.ones(len(x))}).join(pd.DataFrame(x))
	MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
	
	# Note if you don't want to use a DataFrame replace the two lines above with
	# newX = np.append(np.ones((len(x),1)), x, axis=1)
	# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
	
	var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
	sd_b = np.sqrt(var_b)
	ts_b = coef/sd_b
	
	p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
	
	sd_b = np.round(sd_b,10)
	ts_b = np.round(ts_b,10)
	p_values = np.round(p_values,10)
	coef = np.round(coef,10)
	col_names = ['intercept_'] + list(set(x.columns))
	
	df_pValue = pd.DataFrame()
	df_pValue["Feature"],df_pValue["Coefficients"],df_pValue["Standard Errors"],df_pValue["t-statistics"],df_pValue["p-Value"] = [col_names, coef, sd_b, ts_b, p_values]
		
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')
	
	return model, df_pValue
	
def my_API_XGBoost(x, y, x_val, y_val, param={"learning_rate":1.0,
											  "n_estimators":100, 
											  "max_depth":10, 
											  "colsample_bytree":1.0,
											  "colsample_bylevel":1.0,
											  "random_state":66}):
	
	'''
	Date		:	04/20/2018 
	
	Description	:	Apply XGBoost model
	
	Parameters	:	x - Dataframe
						Training dataset with only independent variables
					y - Dataframe 
						Training dataset with only the dependent variable
					x_val - Dataframe
						Testing dataset with only independent variables
					y_val - Dataframe 
						Testing dataset with only the dependent variable
					param - Dictionary (optional)
							Parameters used for the model. Check the link below for details.
							http://xgboost.readthedocs.io/en/latest/python/python_api.html
					
	Return		:	model object
					dataframe
	
	Example		:	N/A
	'''
	
	start = time.time()
	print("#### Parameters for XGB ####")
	print("learning_rate: %0.05f" % param["learning_rate"])
	print("n estimators: %d" % param["n_estimators"])
	print("max depth: %d" % param["max_depth"])
	print("colsample_bytree: %0.05f" % param["colsample_bytree"])
	print("colsample_bylevel: %0.05f" % param["colsample_bylevel"])
	print("###########################\n")
	
	xgboost = XGBClassifier(learning_rate=param["learning_rate"], 
							n_estimators=param["n_estimators"], 
							max_depth=param["max_depth"],
							colsample_bytree=param["colsample_bytree"], 
							colsample_bylevel=param["colsample_bylevel"],
							seed=param["random_state"],
							subsample=0.9, 
							gamma=0, 
							min_child_weight=1,
							objective ='binary:logistic',
							base_score=0.5,
							scale_pos_weight=1, 
							nthread=50)
	model = xgboost
	model.fit(x, y, eval_set=[(x_val, y_val)], eval_metric='logloss', verbose=True, early_stopping_rounds=100)
	
	# feature importance
	importance=model.booster().get_fscore()
	importance=sorted(importance.items(),key=operator.itemgetter(1))
	df_importance = pd.DataFrame(importance, columns=['feature', 'splits'])
	df_importance=df_importance.sort(columns='splits',ascending=False)
	df_importance['fscore'] = df_importance['splits'] / df_importance['splits'].sum()
	
	# plot top 20 important features
	df_importance[df_importance.index < 20].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
	plt.xlabel('importance value')
	plt.ylabel('feature')
	plt.gca().invert_yaxis()
	plt.show()
	
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')
	return model, df_importance

def my_sklearn_dtree(x, y, param = {"max_features":"auto", 
									"max_depth":10, 
									"min_samples_split":0.001,
									"min_samples_leaf":0.0001,
									"random_state":66}):
	
	'''
	Date		:	04/20/2018 
	
	Description	:	Apply decision tree model
	
	Parameters	:	x - Dataframe
						Training dataset with only independent variables
					y - Dataframe 
						Training dataset with only the dependent variable
					param - Dictionary (optional)
							Parameters used for the model. Check the link below for details.
							http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
					
	Return		:	model object
					dataframe
	
	Example		:	N/A
	'''
	
	start = time.time()
	print("#### Parameters for DTree ####")
	print("max features: %s" % param["max_features"])
	print("max depth: %d" % param["max_depth"])
	print("min sample split: %0.05f" % param["min_samples_split"])
	print("min sample leaf: %0.05f" % param["min_samples_leaf"])
	print("###########################\n")
	
	# apply decision tree model
	decision_tree = DecisionTreeClassifier(max_features=param["max_features"],
										   max_depth=param["max_depth"],
										   min_samples_split=param["min_samples_split"], 
										   min_samples_leaf=param["min_samples_leaf"],
										   random_state=66)
	
	imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
	model = Pipeline([('imputation', imp),('Decision Tree', decision_tree)])
	model.fit(x, y)
	
	# feature importance
	df_importance = pd.DataFrame({'feature':x.columns, 'importance':np.round(model.named_steps['Decision Tree'].feature_importances_, 6)})
	df_importance = df_importance.sort('importance',ascending=False)
	df_importance = df_importance.reset_index()
	df_importance = df_importance.drop(['index'], axis=1)
	
	# plot top 20 important features
	df_importance[df_importance.index < 20].plot(kind='barh', x='feature', y='importance', legend=False, figsize=(6, 10))
	plt.xlabel('importance value')
	plt.ylabel('feature')
	plt.gca().invert_yaxis()
	plt.show()
	
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')
	
	return model, df_importance