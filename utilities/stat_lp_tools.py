# File		: stat_lp_tools.py

"""
This module contains necessary helper functions to build the Cplex lp in gf.py
"""

import numpy as np
from utilities.distance_metrics import L2Distance
from sklearn.neighbors import NearestNeighbors

def cost_function(dataset, centres):

	"""
	Given the dataset and cluster centres, this function returns the distance between all samples and cluster centres and returns the lp objective coefficients
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
		centres - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
	Output:
		objective - [ num_variables ] 1D numpy matrix containing weights for corresponding variables
	"""

	transposed = np.expand_dims(dataset, axis = 1)
	distance = np.power(transposed - centres, 2).sum(axis = 2)
	objective = distance.reshape(-1)		# Row major reshaping

	return objective

def prepare_to_add_variables(dataset, centres):

	"""
	Assumes TV distance as statistical distance metric. 

	Takes as input the dataset and cluster centres proposed by some clustering algorirthm and returns the weight vectors
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
		centres - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
	Output:
		objective - [ num_variables ] 1D numpy array containing the weight of different variables
		lower_bound - [ num_variables ] 1D array specifying the minimum value of a variable
		upper_bound - [ num_variables ] 1D array specifying the maximum value of a variable
		variable_names - [ num_variables ] 1D sting list consisting of the names of variables 
		P - Contains the id of the corresponding variables

	Details:
		P_i_k is a real probability variable that denotes the weight for connecting ith sample to kth clusterrr
	"""

	num_samples = len(dataset)
	num_centres = len(centres)
	_id = 0

	# P array contains the id of the corresponding variables
	P = np.zeros([num_samples, num_centres]).astype(int)

	probability_variables = []
	for _point in range(num_samples):
		for _centre in range(num_centres):

			probability_variables.append("P_{point}_{centre}".format(
					point = _point,
					centre = _centre
				))
			# Keep track of P_i_j's position in the lp variable vector 
			P[_point][_centre] = _id
			_id += 1

	variable_names = probability_variables

	# Setting lower bound = 0 and upper bound = 1 for all the variables
	num_variables = len(variable_names)
	lower_bound = [0 for i in range(num_variables)]
	upper_bound = [1 for i in range(num_variables)]

	# Computing the coefficients for objective function
	objective = cost_function(dataset, centres)

	return objective, lower_bound, upper_bound, variable_names, P

def prepare_to_add_constraints(dataset, centres, P, groups, alpha, beta):

	"""
	Assumes TV distance as statistical distance metric. 

	Takes as input the dataset and cluster centres proposed by some clustering algorirthm and returns the constraint details
	Input:
		dataset 	 - [ num_samples x dim ] 2D numpy matrix containing the dataset
		centres 	 - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
		P 	 	 	 - Flattened variable id for P variables defined earlier
		groups		 - [ num_clusters x num_categories ] 2D numpy array partitioning the entire dataset into different protected groups 
					   (2D because a single point can belong to multiple protected groups)
		alpha, beta	 - Constants in statistical fairness constraint
	Output:
		rhs 		- 1D numpy array containing the constants for every equation
		senses 		- a list of strings that identifies whether the corresponding constraint is
    	         	  an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
		row_names 	- a list of string corresponding to the name of the constraint
		coefficients- Three tuple containing the row number, column number and the value of the constraint matrix 
	
	"""

	num_samples = len(dataset)
	num_centres = len(centres)
	num_protected_features = len(alpha)
	assert len(alpha) == len(beta), "Alpha and Beta must have the same dimension"

	rhs = []
	senses = []
	row_names = []
	coefficients = []
	eqn_id = 0				# Denotes the id of the constraint being processed currently

	# Constraint type 1: Summation of P values over all clusters = 1 for each sample
	for point in range(num_samples):
		
		rhs.append(1.0)
		senses.append("E")
		row_names.append("Total_probability_{pt}".format(
				pt = point
			))
		
		for centre in range(num_centres):
			coefficients.append((eqn_id, int(P[point][centre]), 1))

		eqn_id += 1

	# Constraint type 2: Lower bound the fractional representation by beta values
	for protected_feature in range(num_protected_features):
		for protected_group in range(len(beta[protected_feature])):
			for centre in range(num_centres):

				# Ignore when a protected group has no constraints
				if((groups[:, protected_feature] == protected_group).sum() == 0):
					continue

				# Lower bound - beta[i][j] * sum(p[C][k]) <= sum(p[c][k]) where c = points within the group and C is the set of all points
				rhs.append(0)
				senses.append("L")
				row_names.append("{eqn_id}_Lower_Bound_{protected_feature}_{protected_group}_{centre}".format(
						eqn_id = eqn_id,
						protected_feature = protected_feature,
						protected_group = protected_group,
						centre = centre
					))
				for _point in range(num_samples):
					if(groups[_point][protected_feature] == protected_group):
						coefficients.append((eqn_id, int(P[_point][centre]), beta[protected_feature][protected_group] - 1))
					else:
						coefficients.append((eqn_id, int(P[_point][centre]), beta[protected_feature][protected_group]))
				eqn_id += 1

	# Constraint type 3: Upper bound the fractional representation by alpha values
	for protected_feature in range(num_protected_features):
		for protected_group in range(len(alpha[protected_feature])):
			for centre in range(num_centres):

				# Ignore when a protected group has no constraints
				if((groups[:, protected_feature] == protected_group).sum() == 0):
					continue

				# Upper bound - sum(p[c][k]) <= alpha[i][j] * sum(p[C][k]) where c = points within the group and C is the set of all points
				rhs.append(0)
				senses.append("L")
				row_names.append("{eqn_id}_Upper_Bound_{protected_feature}_{protected_group}_{centre}".format(
						eqn_id = eqn_id,
						protected_feature = protected_feature,
						protected_group = protected_group,
						centre = centre
					))
				for _point in range(num_samples):
					if(groups[_point][protected_feature] == protected_group):
						coefficients.append((eqn_id, int(P[_point][centre]), 1 - alpha[protected_feature][protected_group]))
					else:
						coefficients.append((eqn_id, int(P[_point][centre]), -alpha[protected_feature][protected_group]))
				eqn_id += 1

	return rhs, senses, row_names, coefficients

if(__name__ == "__main__"):

	pass
