# File	: distance_metrics.py

import numpy as np

"""
Collection of statistical and euclidean distance metric functions
"""

MU_EPS = 1e-9

def CheckForInf(array):

	"""
	Assertion fails if there are nan values in the array
	"""
	
	assert(not np.isnan(array).any())

def LInfDistance(distribution, NORMALISE_STAT):

	"""
	Takes as input a probability distribution over all the input points and returns pairwise L_infinity distance matrix
	Input:
		distribution - [ num_samples x num_clusters ] 2D matrix
	Output:
		distance - [ num_samples x num_samples ] 2D matrix, [i][j] dentotes the TV distance between distibution i and j
	"""

	transposed = np.expand_dims(distribution, axis = 1)
	p_by_q = distribution / transposed
	CheckForInf(p_by_q)
	q_by_p = transposed / distribution
	CheckForInf(q_by_p)
	max_ratio = np.maximum(p_by_q, q_by_p)
	CheckForInf(max_ratio)
	log_max = np.log(max_ratio)
	CheckForInf(log_max)
	distance = np.max(log_max, axis = 2)
	
	if(NORMALISE_STAT):
		distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance) + MU_EPS)

	return distance

def TVDistance(distribution):

	"""
	Takes as input a probability distribution over all the input points and returns pairwise total variation distance matrix
	Input:
		distribution - [ num_samples x num_clusters ] 2D matrix
	Output:
		distance - [ num_samples x num_samples ] 2D matrix, [i][j] dentotes the TV distance between distibution i and j
	"""

	transposed = np.expand_dims(distribution, axis = 1)
	distance = distribution - transposed
	distance = np.abs(distance).sum(axis = 2) / 2

	return distance

def L2Distance(data, NORMALISE_EUCLIDEAN):

	"""
	returns pairwise l2 distance matrix for all the given points
	Input:
		data - [ num_samples x dim ] 2D matrix
	Output:
		distance - [ num_samples x num_samples ] 2D matrix, [i][j] dentotes the l2 distance between vectors i and j
	"""

	transposed = np.expand_dims(data, axis = 1)
	distance = np.power(data - transposed, 2)
	distance = np.power(np.abs(distance).sum(axis = 2), 0.5) 

	if(NORMALISE_EUCLIDEAN):
		distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance) + MU_EPS)

	return distance

if(__name__ == "__main__"):

	pass