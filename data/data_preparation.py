# File	: data_preparation.py

"""
This is the entry point of Data Preparation phase

This script is reads the dataset in .csv format and dumps the processed data into a pickle file.
The pickle file is a tuple of (dataset, groups)
	dataset  - num_samples x num_features 2D numpy array
	groups 	 - num_samples x num_protected_attributes 2D numpy array

This pickle file is read by the algorithms implemented in Algorithm Execution phase
"""


# Script arguments

# [Bank] - uncomment the following lines to generate pkl file for bank dataset
# TYPE = "bank"		
# PROTECTED_GROUPS = ["marital", "default"]
# FIELDS_OF_INTEREST = ["age", "balance", "duration"]
# DELIMITER = ";"


# [adult] - uncomment the following lines to generate pkl file for adult dataset 
# TYPE = "adult"		
# PROTECTED_GROUPS = ["marital-status", "sex"]
# FIELDS_OF_INTEREST = ["age", "final-weight", "hours-per-week"]
# DELIMITER = ","

# [creditcard] - uncomment the following lines to generate pkl file for creditcard dataset
TYPE = "creditcard"		
PROTECTED_GROUPS = ["MARRIAGE", "EDUCATION"]
FIELDS_OF_INTEREST = ["AGE", "LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3",
						"PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2",
							"BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
DELIMITER = ","


# [census1990] - uncomment the following lines to generate pkl file for census1990 dataset
# TYPE = "census1990"	
# PROTECTED_GROUPS = ["dAge", "iSex"]
# FIELDS_OF_INTEREST = ["dAncstry1", "dAncstry2", "iAvail", "iCitizen", "iClass", "dDepart", 
# 						"iFertil", "iDisabl1", "iDisabl2", "iEnglish", "iFeb55", "dHispanic", "dHour89"]
# DELIMITER = ","


# [diabetes] - uncomment the following lines to generate pkl file for diabetes dataset
# TYPE = "diabetes"		
# PROTECTED_GROUPS = ["gender", "race"]
# FIELDS_OF_INTEREST = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
# DELIMITER = ","


INPUT_FILE = "./raw/" + TYPE + ".csv" 
OUTPUT_FOLDER = "./processed"
MAX_SAMPLES = 1000
NUM_TRIALS = 10
RANDOM_STATE = 0
NORMALISE = True
VARIABLE_SIZE = True
SIZE = [500, 1000, 2000, 3000, 4000]		# Different sizes of the dataset to be subsampled - for timing calculation

# Library imports

import pickle
import numpy as np
import os
from ReaderClasses.Reader import DataReader

if(__name__ == "__main__"):

	dataset = DataReader(INPUT_FILE, FIELDS_OF_INTEREST, PROTECTED_GROUPS, DELIMITER)

	dataset, groups = dataset.CleanData(NORMALISE)

	# Create a file named TYPE.pkl in OUTPUT_FOLDER path
	OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, TYPE)
	# Remove the file if it exists already so that it can be freshly generated
	os.system("rm -rf " + OUTPUT_FOLDER)
	# Create folders if OUTPUT_FOLDER path doesn't exist
	os.makedirs(OUTPUT_FOLDER)
	# Set seed value so that experiment results are consistent
	np.random.seed(RANDOM_STATE)

	# Subsample the dataset NUM_TRIALS many times
	for i in range(1, NUM_TRIALS + 1):

		# Shuffle the dataset
		idx = [i for i in range(len(dataset))]
		np.random.shuffle(idx)
		dataset = dataset[idx]
		groups = groups[idx]

		# Shrink the dataset size to MAX_SAMPLES - equivalent to subsampling uniformly at random
		_dataset = dataset[:MAX_SAMPLES]
		_groups = groups[:MAX_SAMPLES]

		# Dump the cleaned dataset 
		print("\nWriting dataset and protected groups " + str(i) + " to pkl file (" + OUTPUT_FOLDER + ")...")
		file = open(OUTPUT_FOLDER + "/" + TYPE + "_" + str(i) + ".pkl", "wb")
		pickle.dump((_dataset, _groups), file)
		file.close()
		print("Write complete!")

	if(VARIABLE_SIZE):

		dataset = DataReader(INPUT_FILE, FIELDS_OF_INTEREST, PROTECTED_GROUPS, DELIMITER)
		dataset, groups = dataset.CleanData(NORMALISE)
		# Create folder where output files are stored
		OUTPUT_FOLDER = OUTPUT_FOLDER + "_variable_size"
		# Remove the file if it exists already so that it can be freshly generated
		os.system("rm -rf " + OUTPUT_FOLDER)
		# Create folders if OUTPUT_FOLDER path doesn't exist
		os.makedirs(OUTPUT_FOLDER)

		for s in SIZE:

			# Shuffle the dataset
			idx = [i for i in range(len(dataset))]
			np.random.shuffle(idx)
			dataset = dataset[idx]
			groups = groups[idx]

			_dataset = dataset[:s]
			_groups = groups[:s]

			# Dump the cleaned dataset 
			print("\nWriting dataset and protected groups size = " + str(s) + " to pkl file (" + OUTPUT_FOLDER + ")...")
			file = open(OUTPUT_FOLDER + "/" + TYPE + "_" + str(s) + ".pkl", "wb")
			pickle.dump((_dataset, _groups), file)
			file.close()
			print("Write complete!")