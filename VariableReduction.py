def my_corr(df, method, abs=False):
	'''
	Date		:	04/18/2018 
	
	Description	:	Calculate the correlation coefficient for all possible pairs
	
	Parameters	:	df - Dataframe
					method - string	{‘pearson’, ‘kendall’, ‘spearman’}
					abs - Boolean (optinal)
						Remove the negative sign
	
	Return		:	A dataframe contains all possible pairs and corresponding correlation coefficient
	
	Example		:	my_corr(df_MS_600K_1MM, "pearson", abs=False)
	'''
	
	start = time.time()
	
	if abs:
		df_corr = df.corr(method=method, min_periods=1).abs().unstack()
	else:
		df_corr = df.corr(method=method, min_periods=1).unstack()
	
	# Get diagonal and lower triangular pairs of correlation matrix; Drop duplicated pairs
	pairs_to_drop = set()
	cols = df.columns
	for i in range(0, df.shape[1]):
		for j in range(0, i+1):
			pairs_to_drop.add((cols[i], cols[j]))
	df_corr = pd.DataFrame(df_corr.drop(labels=pairs_to_drop)).sort(ascending=False)
	
	end = time.time()
	print('time elapsed: ' + str(end - start) + ' seconds')
	return df_corr
	
def cal_woe_iv(df, var_x, var_y):
    for j in var_y:
        bins_optimal_df = pd.DataFrame()
        for i in var_x:

            x2=df

            df2 = x2[np.isnan(x2[i])==False]

            y = df2[[j]]
            x = df2[[i]]

            clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=2, min_samples_leaf=500, 
                                         random_state=0, max_leaf_nodes=20,min_impurity_split=0.001)

            y = y.fillna(0)
            clf.fit(x,y)

            from sklearn.tree import _tree

            #print "Bin Code"
            #print " "

            def tree_to_code(tree, feature_names, target):
                tree_ = tree.tree_
                feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature]
                print "def tree({}):".format(", ".join(feature_names))

                def recurse(node, depth):
                    indent = "  " * depth
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:
                        name = feature_name[node]
                        threshold = tree_.threshold[node]
                        print "{}if {} <= {}:".format(indent, name, threshold)
                        recurse(tree_.children_left[node], depth + 1)
                        print "{}else:  # if {} > {}".format(indent, name, threshold)
                        recurse(tree_.children_right[node], depth + 1)
                    else:
                        print "{}return {}".format(indent, tree_.value[node])

                #recurse(0, 1)

                cuts = []
                cuts.append(tree_.threshold[0])

                def recurse2(node):
                    if tree_.feature[node] != _tree.TREE_UNDEFINED:

                        recurse2(tree_.children_left[node])
                        cuts.append(tree_.threshold[node])

                        recurse2(tree_.children_right[node])
                        cuts.append(tree_.threshold[node])

                recurse2(0)

                cuts = np.unique(np.array(cuts,dtype=np.float32))
                cuts = np.sort(cuts)

                i=0
                df_cuts =pd.DataFrame()
                temp = pd.Series()
                temp2 = pd.concat([x,y], axis=1)

                #min bin
                df_cuts['low'] = temp2[[feature_names[0]]].min()
                df_cuts['high'] = cuts[i]
                df_cuts['total'] = temp2[(temp2[feature_names[0]]>=temp2[feature_names[0]].min()) & (temp2[feature_names[0]]<cuts[i])][target[0]].count()
                df_cuts['bads'] = temp2[(temp2[feature_names[0]]>=temp2[feature_names[0]].min()) & (temp2[feature_names[0]]<cuts[i])][target[0]].sum()
                df_cuts['goods'] = df_cuts.total - df_cuts.bads
                df_cuts['bad_rate'] = (df_cuts.bads / df_cuts.total)*100

                # in b/w bins

                while i < cuts.size - 1:

                    temp['low'] = cuts[i]
                    temp['high'] =cuts[i+1]
                    temp['total'] = temp2[(temp2[feature_names[0]]>=cuts[i]) & (temp2[feature_names[0]]<cuts[i+1])][target[0]].count()
                    temp['bads'] = temp2[(temp2[feature_names[0]]>=cuts[i]) & (temp2[feature_names[0]]<cuts[i+1])][target[0]].sum()
                    temp['goods'] = temp.total - temp.bads
                    temp['bad_rate'] = (temp.bads / temp.total)*100
                    df_cuts=df_cuts.append(temp, ignore_index=True)
                    i=i+1

                # max bin
                temp['low'] = cuts[cuts.size - 1]
                temp['high'] = temp2[feature_names[0]].max()
                temp['total'] = temp2[temp2[feature_names[0]]>=cuts[cuts.size - 1]][target[0]].count()
                temp['bads'] = temp2[temp2[feature_names[0]]>=cuts[cuts.size - 1]][target[0]].sum()
                temp['goods'] = temp.total - temp.bads
                temp['bad_rate'] = (temp.bads / temp.total)*100
                df_cuts=df_cuts.append(temp, ignore_index=True)

                # missing bin
                temp['low'] = np.nan
                temp['high'] = np.nan
                temp['total'] = x2[np.isnan(x2[feature_names[0]])== True][target[0]].count()
                temp['bads'] = x2[np.isnan(x2[feature_names[0]])== True][target[0]].sum()
                temp['goods'] = temp.total - temp.bads
                temp['bad_rate'] = (temp.bads / temp.total)*100
                df_cuts=df_cuts.append(temp, ignore_index=True)

                #print " "
                #print "Optimal Bins"
                #print " "
                df_cuts['IV_bin'] = ((df_cuts['goods'] / df_cuts['goods'].sum()) - (df_cuts['bads'] / df_cuts['bads'].sum()))  * np.log((df_cuts['goods'] / df_cuts['goods'].sum())/((df_cuts['bads']+0.01)/ df_cuts['bads'].sum())) 
                df_cuts['IV'] = df_cuts['IV_bin'].sum()
                #df_cuts['IV'] = df_cuts[np.isnan(df_cuts.low)==False]['IV_bin'].sum()
                df_cuts['variable_name'] = feature_names[0]
                #print df_cuts
                return df_cuts

            optimal_bins = tree_to_code(clf,[i],[j])
            bins_optimal_df = pd.concat([bins_optimal_df,optimal_bins])
        return bins_optimal_df

from sklearn.cluster import FeatureAgglomeration
ward_cluster = FeatureAgglomeration(n_clusters=50, affinity='euclidean', linkage='ward', compute_full_tree=bool)
ward_cluster.fit(df_x.fillna(df_x.mean()), df_y)

mann_cluster = FeatureAgglomeration(n_clusters=50, affinity='l1', linkage='complete')
mann_cluster.fit(df_x.fillna(df_x.mean()), df_y)

df_cluster = pd.DataFrame()
df_cluster['feature'] = pd.Series(df_x.columns)
df_cluster['ward_cluster'] = pd.Series(ward_cluster.labels_)
df_cluster['mann_cluster'] = pd.Series(mann_cluster.labels_)

df_cluster = df_cluster.merge(df_iv_pbg, left_on='feature', right_on='variable_name', how='left')


				    
				    
