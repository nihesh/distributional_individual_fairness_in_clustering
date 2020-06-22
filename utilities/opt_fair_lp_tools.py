# File		: opt_fair_lp_tools.py

"""
This module contains necessary helper functions to build the Cplex lp in opt_if.py
"""

import numpy as np
from utilities.distance_metrics import L2Distance
from sklearn.neighbors import NearestNeighbors

def cost_function(dataset, num_variables):

	"""
	Given the dataset and cluster centres, this function returns the distance between all samples and cluster centres and returns the lp objective coefficients
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
		centres - [ num_clusters x dim ] 2D numpy matrix containing the cluster centres
	Output:
		objective - [ num_variables ] 1D numpy matrix containing weights for corresponding variables
	"""

	transposed = np.expand_dims(dataset, axis = 1)
	distance = np.power(transposed - dataset, 2).sum(axis = 2)
	distance = distance.reshape(-1)		# Row major reshaping

	zero_vector = np.zeros(num_variables - len(distance))	# Pad the rest with 0s. They don't contribute to cost

	objective = np.concatenate([distance, zero_vector], axis = 0)

	return objective

def prepare_to_add_variables(dataset):

	"""
	Assumes TV distance as statistical distance metric. 

	Takes as input the dataset and returns the objective function weight vector
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
	Output:
		objective - [ num_variables ] 1D numpy array containing the weight of different variables
		lower_bound - [ num_variables ] 1D array specifying the minimum value of a variable
		upper_bound - [ num_variables ] 1D array specifying the maximum value of a variable
		variable_names - [ num_variables ] 1D sting list consisting of the names of variables 
		P, C, Y - P, C and Y array contain the id of the corresponding variables

	Details:
		P_i_k is a real probability variable that denotes the weight for connecting ith sample to kth clusterrr
		C_i_j_k is a intermediate variable that is an upper bound on | P_i_k - P_j_k | - A hack to convert non linear abs constraint to a linear constraint
		Y_k is the fraction by which centre k is opened
	"""

	num_samples = num_centres = len(dataset)
	_id = 0

	# P and C array contain the id of the corresponding variables
	P = np.zeros([num_samples, num_centres]).astype(int)
	C = np.zeros([num_samples, num_samples, num_centres]).astype(int)
	Y = np.zeros([num_samples]).astype(int)

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

	abs_constraint_variables = []
	for _point1 in range(num_samples):
		for _point2 in range(_point1 + 1, num_samples):
			for _centre in range(num_centres):

				abs_constraint_variables.append("C_{point1}_{point2}_{centre}".format(
						point1 = _point1,
						point2 = _point2,
						centre = _centre
					))
				# Keep track of C_i_j_k's position in lp variable vector
				C[_point1][_point2][_centre] = C[_point2][_point1][_centre] = _id
				_id += 1

	cluster_frac_variables = []
	for centre in range(num_centres):

		cluster_frac_variables.append("Y_{centre}".format(
				centre = centre,
			))

		# Keep track of Y_k's position in lp variable vector
		Y[centre] = _id
		_id += 1

	# Concatenating the names of both the types of variables
	variable_names = probability_variables + abs_constraint_variables + cluster_frac_variables

	# Setting lower bound = 0 and upper bound = 1 for all the variables
	num_variables = len(variable_names)
	lower_bound = [0 for i in range(num_variables)]
	upper_bound = [1 for i in range(num_variables)]

	# Computing the coefficients for objective function
	objective = cost_function(dataset, num_variables)

	return objective, lower_bound, upper_bound, variable_names, P, C, Y

def prepare_to_add_constraints(dataset, k, n_neighbours, P, C, Y, fairness_type):

	"""
	Assumes TV distance as statistical distance metric. 

	Takes as input the dataset and returns the constraint details
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing the dataset
		k 		- Number of cluster centres to open
		n_neighbours - Number of neighbours to consider for enforcing fairness constraint
		P, C, Y	- Flattened variable id for P, C and Y variables defined earlier
		fairness_type - 0 <- metric fairness 1 <- local fairness
	Output:
		rhs - 1D numpy array containing the constants for every equation
		senses - a list of strings that identifies whether the corresponding constraint is
    	         an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
		row_names: a list of string corresponding to the name of the constraint
		coefficients: Three tuple containing the row number, column number and the value of the constraint matrix 
	
	"""

	# Build ball_tree for kNN queries
	knn = NearestNeighbors(n_neighbors = n_neighbours + 1, algorithm = "ball_tree").fit(dataset)	# One more than n_neighbours as query point itself is a neighbour which is ignored
	_, neighbours = knn.kneighbors(dataset)
	neighbours = neighbours[:, 1:]			# Remove the first nearest neighbour which is itself

	num_samples = len(dataset)
	num_centres = num_samples 				# Every sample point is now a centre, which is partially opened (denoted by Y)

	rhs = []
	senses = []
	row_names = []
	coefficients = []
	eqn_id = 0				# Denotes the id of the constraint being processed currently

	distance = L2Distance(dataset, True)

	# Use alternate fairness measure
	if(fairness_type == 1):

		# Create selection mask - selection[i][j] is true if j is one of top 'n_neighbours' neighbours of i.
		selection = np.zeros([num_samples, num_samples]).astype(bool)
		row_id = np.asarray([i for i in range(num_samples) for j in range(n_neighbours)]).reshape(-1)
		selection[row_id, neighbours.reshape(-1)] = True

		distance[np.logical_not(selection)] = 0
		distance = (distance) / (np.max(distance, axis = 1)).reshape(-1, 1)

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

	# Constraint type 2: Lower and upper bound the abs values using their corresponding C variables
	for _point1 in range(num_samples):
		for _point2 in neighbours[_point1]:
			for centre in range(num_centres):

				# Upper bound - P[i][k] - P[j][k] <= C[i][j][k]
				rhs.append(0)
				senses.append("L")
				row_names.append("{eqn_id}_Upper_Bound_{pt1}_{pt2}_{centre}".format(
						eqn_id = eqn_id,
						pt1 = _point1,
						pt2 = _point2,
						centre = centre
					))
				coefficients.append((eqn_id, int(P[_point1][centre]), 1))
				coefficients.append((eqn_id, int(P[_point2][centre]), -1))
				coefficients.append((eqn_id, int(C[_point1][_point2][centre]), -1))

				eqn_id += 1

				# Lower_bound - P[i][k] - P[j][k] >= -C[i][j][k]
				rhs.append(0)
				senses.append("G")
				row_names.append("{eqn_id}_Lower_bound_{pt1}_{pt2}_{centre}".format(
						eqn_id = eqn_id,
						pt1 = _point1,
						pt2 = _point2,
						centre = centre
					))
				coefficients.append((eqn_id, int(P[_point1][centre]), 1))
				coefficients.append((eqn_id, int(P[_point2][centre]), -1))
				coefficients.append((eqn_id, int(C[_point1][_point2][centre]), 1))

				eqn_id += 1

	# Constraint type 3: Add fairness constraints in terms of C variables - trick to make mod constraints linear
	for _point1 in range(num_samples):
		for _point2 in neighbours[_point1]:

			rhs.append(2 * distance[_point1][_point2])		# Multiply by 2 to account for the division in TV distance expression
			senses.append("L")
			row_names.append("{eqn_id}_Fainess_{pt1}_{pt2}".format(
					eqn_id = eqn_id,
					pt1 = _point1,
					pt2 = _point2
				))

			for centre in range(num_centres):
				coefficients.append((eqn_id, int(C[_point1][_point2][centre]), 1))

			eqn_id += 1

	# Constraint type 4: Sum of all Y_i = k (exactly k centres are opened)
	rhs.append(k)		
	senses.append("E")
	row_names.append("Open k centres")
	for centre in range(num_centres):
		coefficients.append((eqn_id, int(Y[centre]), 1))
	eqn_id += 1

	# Constraint type 5: No sample is assigned to a centre more than the amount by which it's opened. P_i_k <= Y_k
	for point in range(num_samples):
		for centre in range(num_centres):

			rhs.append(0)
			senses.append("L")
			row_names.append("{eqn_id}_Cluster_Capacity_{point}_{centre}".format(
					eqn_id = eqn_id,
					point = point,
					centre = centre
				))
			coefficients.append((eqn_id, int(P[point][centre]), 1))
			coefficients.append((eqn_id, int(Y[centre]), -1))

			eqn_id += 1

	return rhs, senses, row_names, coefficients

if(__name__ == "__main__"):

	pass
