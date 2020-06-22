# File		: fair_stat_lp_tools.py

"""
This module contains necessary helper functions to build the Cplex lp in opt_cf.py
"""

import numpy as np
from utilities.distance_metrics import L2Distance
from sklearn.neighbors import NearestNeighbors

def find_pairs_with_common_groups(groups):

	"""
	Returns a list of pairs of indices i, j such that i and j belong to the same protected group
	Input:
		groups		 - [ num_clusters x num_categories ] 2D numpy array partitioning the entire dataset into different protected groups 
					   (2D because a single point can belong to multiple protected groups)
	Output:
		good_pairs 	 - List of pairs
	"""

	# Convert groups into a 2D numpy array in case it is 1D
	if(len(groups.shape) == 1):
		groups = groups.reshape(-1, 1)

	num_samples = groups.shape[0]
	num_protected_features = groups.shape[1]

	# selection[i][j] is true if i and j have some protected group in common
	selection = np.zeros([num_samples, num_samples]).astype(bool)

	# Iterate over protected feautres
	for i in range(num_protected_features):
		protected_groups = list(set(groups[:, i]))
		
		# Iterate over protected groups
		for group in protected_groups:
			mask = np.where(groups[:, i] == group)[0]
			mask = [np.tile(mask, len(mask)), np.repeat(mask, len(mask))]
			selection[mask[0], mask[1]] = True 	# selection i, j = True if and only if i, j belong to some same protected group

	for i in range(num_samples):
		selection[i][i] = False

	good_pairs = []
	# Iterate over items in selection matrix and add it to good pairs if corresponding (i, j) is set to True
	for i in range(num_samples):
		for j in range(num_samples):
			if(selection[i][j]):
				good_pairs.append((i, j))

	return good_pairs

def prepare_to_add_individual_fairness_constraints(dataset, k, groups, P, C, Y, eqn_id, n_neighbours, fairness_type):

	"""
	Assumes TV distance as statistical distance metric. 

	Takes as input the dataset and cluster centres proposed by some clustering algorirthm and returns the constraint details
	Input:
		dataset 	 - [ num_samples x dim ] 2D numpy matrix containing the dataset
		k 			 - Number of cluster centres to open
		groups		 - [ num_clusters x num_categories ] 2D numpy array partitioning the entire dataset into different protected groups 
					   (2D because a single point can belong to multiple protected groups)
		P, C and Y   - Flattened variable id for P, C and Y variables defined earlier
		eqn_id		 - Next equation number to be used - CPLEX related variable
		n_neighbours- Number of nearest neighbours to enforce fairness constraint on - given that two points belong to some common protected group
		fairness_type - 0 <- metric fairness 1 <- local fairness
	Output:
		rhs 		- 1D numpy array containing the constants for every equation
		senses 		- a list of strings that identifies whether the corresponding constraint is
    	         	  an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
		row_names 	- a list of string corresponding to the name of the constraint
		coefficients- Three tuple containing the row number, column number and the value of the constraint matrix 
	
	"""

	# Convert groups to a 2D array in case there is only one protected group
	if(len(groups.shape) == 1):
		groups = groups.reshape(-1, 1)

	num_samples = len(dataset)
	num_centres = num_samples
	num_protected_features = groups.shape[1]

	# Build ball_tree for kNN queries
	knn = NearestNeighbors(n_neighbors = n_neighbours + 1, algorithm = "ball_tree").fit(dataset)	# One more than n_neighbours as query point itself is a neighbour which is ignored
	_, neighbours = knn.kneighbors(dataset)
	neighbours = neighbours[:, 1:]			# Remove the first nearest neighbour which is itself

	# Create selection mask
	selection = np.zeros([num_samples, num_samples]).astype(bool)
	row_id = np.asarray([i for i in range(num_samples) for j in range(n_neighbours)]).reshape(-1)
	selection[row_id, neighbours.reshape(-1)] = True

	# Find good pairs, goodness defined within the function
	good_pairs = find_pairs_with_common_groups(groups)

	rhs = []
	senses = []
	row_names = []
	coefficients = []
	
	distance = L2Distance(dataset, True)	# Compute pairwise distances between points

	# Use alternate fairness measure
	if(fairness_type == 1):

		distance[np.logical_not(selection)] = 0
		distance = (distance) / (np.max(distance, axis = 1)).reshape(-1, 1)

	# Constraint type 2: Lower and upper bound the abs values using their corresponding C variables
	for _point1, _point2 in good_pairs:
		
		# If good pair and point2 is not within knn of point 1, don't enforce individual fairness
		if(not selection[_point1][_point2]):
			continue

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
	for _point1, _point2 in good_pairs:

		# If good pair and point2 is not within knn of point 1, don't enforce individual fairness
		if(not selection[_point1][_point2]):
			continue

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
