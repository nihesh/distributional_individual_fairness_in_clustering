# File	: helpers.py

"""
Implementation of helper functions for Reader.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def RemoveNonNumerics(data, protected_groups):

	"""
	[Helper Function]
	Takes as input a dataframe and removes categorical attributes and constant attributes along with protected_groups
	Input 	: Dataframe with categorical data and list of protected_groups
	Output	: Dataframe without categorical data
	"""

	# Select only those colums with integer attributes
	data = data.select_dtypes(include = ['int64'])
	# Remove attributes which take the same value across all the records
	data = data.loc[:, (data != data.iloc[0]).any()]
	
	# Drop columns in protected_groups
	to_drop = []
	for group in protected_groups:
		if(group in data.columns):
			to_drop.append(group)
	data = data.drop(columns = to_drop)
	
	return data

def ReadRawData(src, delimiter):

	"""
	[Helper Function]
	Reads the raw subsampled csv filed specified in src and returns a dataframe
	Input   : source path and delimiter that separates the attributes
	Output  : Loaded data frame for raw data specified in src in csv format
	"""

	data = pd.read_csv(src, sep = delimiter)
	return data

def CenterData(data):

	"""
	[Helper Function]
	Takes as input a 2D data matrix and converts it into zero mean unit variance on every feature axis 
	Input 	: 2D data matrix
	Output	: Normalised 2D data matrix
	"""

	scaler = StandardScaler(copy = True, with_mean = True, with_std = True)
	data = data.astype(float)
	data = scaler.fit_transform(data)

	return data

def ExtractProtectedGroups(data, protected_attributes):

	"""
	[Helper Function]
	Takes as input the data frame and the list of protected attributes, checks their validity and
	converts string classes to 0 to C - 1 where C is the number of classes.
	Input 	: 
		data - 2D data frame
		protected_attributes - List of attribute names defining the protected classes
	Output	:
		Integer labels for each protected group - 2D numpy matrix [ num_samples x num_protected_attributes ]
	"""

	# Check if protected groups are valid
	ok = True
	bad_name = []
	for group in protected_attributes:
		if(group not in data.columns):
			ok = False
			bad_name.append(group)

	assert ok, "protected groups in the following list are not found\n" + str(bad_name)

	# Choose only the protected groups
	groups = data[protected_attributes]
	# Print the categorical attributes
	for val in protected_attributes:
		print("Categorical values for " + val)
		print(set(data[val]))

	# Convert string data to running numbers from 0 to C - 1 where C is the number of distinct categorical classes inside a protected_group field
	le = LabelEncoder()
	groups[protected_attributes] = groups[protected_attributes].apply(lambda col: le.fit_transform(col))	# Ignore warning here, while running the code

	return groups.values

if(__name__ == "__main__"):

	pass