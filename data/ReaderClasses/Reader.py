# File	: ReaderClasses.py

"""
Data reader API
"""

from ReaderClasses.helpers import *

class DataReader():

	def __init__(self, src, fields_of_interest, protected_attributes, delimiter):

		"""
		src 					- path to the csv file
		fields_of_interest 		- Attributes in the csv file which should be preserved. Remaining attributes will be dropped
		protected_attributes		- Attributes in the csv file which define protected classes
		delimiter 				- Delimiter that separates the attributes in the csv file
		"""

		self.src = src
		self.fields_of_interest = fields_of_interest
		self.protected_attributes = protected_attributes
		self.delimiter = delimiter

	def CleanData(self, normalise = False):

		"""
		[Class Method]
		Loads the dataset provided in src and cleans the raw data
		Input 	: normalise - normalises the feature dimension by making it 0 mean and unit variance
		Output	: 2D numpy array containing required data for clustering
		"""

		print("Reading the dataset...")
		# Read data from the disk into a dataframe
		data = ReadRawData(self.src, self.delimiter)
		# Drop columns in the dataframe that doesn't belong to fields_of_interest U protected_attributes
		select = list(set(self.fields_of_interest).union(set(self.protected_attributes)))
		data = data[select]
		print("Data read successfully")

		print("\nAttributes found: ")
		print(data.columns)
		print("\nProtected attributes: ")
		print(self.protected_attributes)

		print("\nChecking if protected attributes are valid and extracting the group labels...")
		# encode every protected group for a protected attribute using integers from 0...C - 1 where C is the number of protected groups
		groups = ExtractProtectedGroups(data, self.protected_attributes)
		print("\nProtected groups extraction successful")
		
		# Remove non numeric colums from the dataframe
		data = RemoveNonNumerics(data, self.protected_attributes)
		print("\nAttributes after removing non numeric features and protected attributes:")
		print(data.columns)
		
		# Remove duplicate entries in the dataset - If duplicates are necessary, add random small noise to make them unique.
		idx = ~data.duplicated()
		data = data[idx]
		groups = groups[idx]
		
		data = data.values

		# Normalisation of attributes in the dataset
		if(normalise):
			print("\nCentering the data to zero mean, unit variance gaussian...")
			data = CenterData(data)
			print("Data centred successfully")

		return (data, groups)
