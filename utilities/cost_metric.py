# File	: cost_metric.py

"""
This module defines the cost functions for feasible solutions to kmeans and other clustering algorithms
"""

import numpy as np

def KMeansCost(data, centres, distribution):

	"""
	Given the dataset, cluster centres and distribution, it computes the expected cost of the solution
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing data samples
		centres - [ num_clusters x dim ] 2D numpy matrix containing cluster centres
		distribution - [ num_samples x num_clusters ] 2D numpy matrix containing distribution over cluster centres for each data sample 
	Output:
		Single floating number, the expect cost of the cluster, per sample 
	"""

	num_samples = data.shape[0]

	data = np.expand_dims(data, axis = 1)
	distance = np.power(data - centres, 2).sum(axis = 2)

	weighted_cost = distance * distribution
	expected_cost = weighted_cost.sum()

	return expected_cost / num_samples

if(__name__ == "__main__"):

	pass
