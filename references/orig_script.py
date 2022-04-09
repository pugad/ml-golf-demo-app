from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

# from IPython.display import Image
# import pydotplus
# import graphviz

def data_collection():

	#create empty data frame
	golf_df = pd.DataFrame()

	# add outlook
	golf_df['Outlook'] = ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 
						 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast',
						 'overcast', 'rainy']

	#add temperature (deg Celsius)
	golf_df['Temperature'] = [33, 30, 28, 10, 16, 12, 17, 24, 14, 22, 26, 21, 16, 22]

	#add humidity (percent)
	golf_df['Humidity'] = [.403, .631, .353, .863, .724, .227, .270,
						  .807, .154, .251, .186, .583, .472, .393]

	#add windy
	golf_df['Windy'] = ['false', 'true', 'false', 'false', 'false', 'true', 'true',
					   'false', 'false', 'false', 'true', 'true', 'false', 'true']

	#finally add play
	golf_df['Play'] = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 
					   'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']


	return(golf_df)

def one_hot_encode(df):

	one_hot_data = pd.get_dummies(df[['Outlook','Windy']])
	return(one_hot_data)

def join_datasets(df1,df2):

	joined = df1[['Temperature','Humidity']].join(df2)
	return(joined)



def train_model(one_hot_df,results_series):

	clf = tree.DecisionTreeClassifier() # <== we're answering yes or no questions


	clf_train = clf.fit(one_hot_df,results_series)


	fig, axes = plt.subplots(nrows = 1,
							ncols = 1,
							figsize = (4,4),
							dpi = 300)

	tree.plot_tree(
					clf_train,
					feature_names = one_hot_df.columns,
					class_names = ['No golf','Golf'],
				filled = True
					)

	# plt.show()
	fig.savefig('trained_model.png')
	
	return(clf_train)



def main():

	df = data_collection()
	one_hot_df = one_hot_encode(df)
	joined_df = join_datasets(df,one_hot_df)


	trained_model = train_model(joined_df,df['Play'].tolist())

	print('Columns: ',joined_df.columns.values)
	
	predict_me = pd.DataFrame({
		'Temperature':[20],
		'Humidity':[0.6],
		'Outlook_overcast':[0],
		'Outlook_rainy':[0],
		'Outlook_sunny':[1],
		'Windy_false':[1],
		'Windy_true':[0]
	})
	prediction = trained_model.predict(predict_me)
	
	print('Prediction: ',prediction)


if __name__ == '__main__':
	main()