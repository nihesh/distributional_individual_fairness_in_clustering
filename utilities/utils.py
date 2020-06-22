# File	: utils.py

from utilities.distance_metrics import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import scipy
from pyemd import emd
import random

"""
Utility functions
"""

def ResetWorkspace():

	"""
	Sets seed values for various RNGs 
	"""

	random.seed(0)
	np.random.seed()

def CheckForInf(array):

	"""
	Assertion fails if there is nan or infinityin the array
	"""

	assert(not np.isnan(array).any())
	assert(not np.isinf(array).any())

def SingleAxisPlot(x_axis, y_axis, save_path, 
					x_label = "x_axis",
					y_label = "y_axis",
					title = ""
				):

	"""
	Draws a normal single y axis plot with the given parameters
	Input:
		x_axis				: [ N ] shaped vector consisting of x axis values
		y_axis 				: [ N ] shaped vector consisting of corresponding y axis values
		save_path			: Directory to which the output file is saved - string
		x_label, y_label 	: Labels of corresponding axes - strings
		title 				: Plot title - string
	Output:
		None
	"""

	plt.clf()
	plt.plot(x_axis, y_axis)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)

	plt.savefig(save_path)
	plt.close("all")

def SecondaryAxisPlot(x_axis, y1_axis, y2_axis, save_path, 
						x_label = "x axis", 
						y1_label = "primary y axis", 
						y2_label = "secondary y axis",
						title = ""
					):

	"""
	Draws a secondary axis plot with the given parameters
	Input:
		x_axis							: [ N ] shaped vector consisting of x axis values
		y1_axis 						: [ N ] shaped vector consisting of corresponding left y axis values
		y2_axis							: [ N ] shaped vector consisting of corresponding right y axis values
		save_path						: Directory to which the output file is saved - string
		x_label, y1_label, y2_label 	: Labels of corresponding axes - strings
		title 							: Plot title - string
	Output:
		None 
	"""

	plt.clf()
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	
	ax1.plot(x_axis, y1_axis, 'g-')
	ax2.plot(x_axis, y2_axis, 'b-')
	ax1.set_xlabel(x_label)
	ax1.set_ylabel(y1_label, color='g')
	ax2.set_ylabel(y2_label, color='b')
	plt.title(title)

	plt.savefig(save_path)
	plt.close("all")

def nearestNeighbourDistribution(dataset, kmeans):

	"""
	Computes the nearest cluster for every point and assigns a probability of 1 to it, 0 elsewhere
	Input:
		dataset - [ num_samples x dim ] matrix containing data to be clustered
		kmeans - sklearn object containing cluster information
	Output:
		Matrix of dim [ num_samples x num_clusters ]
	"""

	labels = kmeans.labels_
	labels = labelToOneHot(labels)

	return labels

def InverseL2Distribution(dataset, kmeans):

	"""
	Computes a distribution inversely proportional to L2 distance from cluster centres, normalised by sum
	Input:
		dataset - [ num_samples x dim ] matrix containing data to be clustered
		kmeans - sklearn object containing cluster information
	Output:
		Matrix of dim [ num_samples x num_clusters ]
	"""

	dataset = np.expand_dims(dataset, axis = 1)
	centres = kmeans.cluster_centers_
	distance = np.power(dataset - centres, 2)
	distance = distance.sum(axis = 2)
	distribution = 1 / distance
	distribution = distribution / np.expand_dims(np.abs(distribution.sum(axis = 1)), 1)

	return distribution

def SoftmaxInvL2Distribution(dataset, kmeans, temperature):

	"""
	Computes a distribution over negative L2 distance from cluster centres, normalised by temperature scaled softmax
	Input:
		dataset - [ num_samples x dim ] matrix containing data to be clustered
		kmeans - sklearn object containing cluster information
	Output:
		Matrix of dim [ num_samples x num_clusters ]
	"""

	dataset = np.expand_dims(dataset, axis = 1)
	centres = kmeans.cluster_centers_
	distance = np.power(dataset - centres, 2)
	distance = distance.sum(axis = 2)
	distribution = (-distance) * temperature

	distribution = distribution - np.max(distribution, axis = 1).reshape(-1, 1)

	# Softmax activation
	distribution = np.exp(distribution)
	CheckForInf(distribution)

	distribution = distribution / np.expand_dims(distribution.sum(axis = 1), axis = 1)
	CheckForInf(distribution)

	return distribution

def computeDistribution(dataset, kmeans, distribution, temperature):

	"""
	Input:
		dataset - [ num_samples x dim ] matrix containing data to be clustered
		kmeans - sklearn object containing cluster information
	Output:
		Probability distribution over cluster centres for each data point
		Matrix of dim [ num_samples x num_clusters ]
	"""

	if(distribution == 0):
		distribution = nearestNeighbourDistribution(dataset, kmeans)
	elif(distribution == 1):
		distribution = InverseL2Distribution(dataset, kmeans)
	elif(distribution == 2):
		distribution = SoftmaxInvL2Distribution(dataset, kmeans, temperature)
	else:
		print("Invalid distribution code")
		exit(0)

	return distribution

def FindStatisticalDistance(distribution, stat_distance_metric, NORMALISE_STAT = False):

	"""
	Takes as input a probability distribution over all the input points and returns pairwise statistical distance matrix
	Input:
		distribution - [ num_samples x num_clusters ] 2D matrix
		stat_distance_metric - code to specify the distance metric to be used
	Output:
		distance - [ num_samples x num_samples ] 2D matrix, [i][j] dentotes the statistical distance between distibution i and j
	"""

	if(stat_distance_metric == 0):
		# TV distance is already between 0 and 1 - normalisation is not required
		distance = TVDistance(distribution)
	elif(stat_distance_metric == 1):
		distance = LInfDistance(distribution, NORMALISE_STAT)
	else:
		print("Invalid statistical distance metric code")
		exit(0)

	return distance

def FindEuclideanDistance(data, euclidean_distance_metric, NORMALISE_EUCLIDEAN = False):

	"""
	Takes as input a set of points and returns pairwise distances between them
	Input:
		data - [ num_samples x num_clusters ] 2D matrix
		euclidean_distance_metric - code to specify the distance metric to be used
	Output:
		distance - [ num_samples x num_samples ] 2D matrix, [i][j] dentotes the statistical distance between distibution i and j
	"""

	if(euclidean_distance_metric == 0):
		distance = L2Distance(data, NORMALISE_EUCLIDEAN)
	else:
		print("Invalid euclidean distance metric code")
		exit(0)

	return distance

def labelToOneHot(label):

	"""
	Converts a given vector of labels to a 2D matrix where each row is a one hot encoded vector
	Input:
		label - [N] dim numpy array containing int64 values from 0 to Classes - 1
	Output:
		[N, C] 2D numpy array
	"""

	N = label.shape[0]
	Classes = np.max(label) + 1
	encoding = np.zeros([N, Classes])
	encoding[np.arange(N), label] = 1

	return encoding

def CountViolations(euclidean_distance, stat_distance):

	"""
	Counts the number of constraints where stat_distance > euclidean_distance
	Input:
		euclidean_distance - [ num_samples x num_samples ] 2D matrix denoting pairwise euclidean distance
		stat_distance - [ num_samples x num_samples ] 2D matrix denoting pairwise statistical distance
	Output:
		(total_pairs, violating_pairs)
		total_pairs - total number of unordered pairs - num_samples choose 2
		violating_pairs - Number of pairs that violate the constraint (unordered) 
	"""

	global MU_EPS

	assert(euclidean_distance.shape == stat_distance.shape)

	num_samples = euclidean_distance.shape[0]
	num_neighbours = euclidean_distance.shape[1]
	total = num_samples * num_neighbours
	# Subtract EPS to ensure numerical stability
	violations = (stat_distance - MU_EPS > euclidean_distance).sum() 	# Counts the number of ordered pairs 
	
	return total, violations

def PercentageViolations(distribution, dataset, n_neighbours = 0, title = "", verbose = False,
			stat_distance_metric = 0, euclidean_distance_metric = 0, 
			normalise_stat = True, normalise_euclidean = True, fairness_type = 0
		):

	"""
	Given the dataset and distribution over the points, this function computers number of violations of Lipschitz constraint
	Input:
		distribution - [ num_samples x num_clusters ] 2D numpy matrix containing distribution over cluster centres for each data sample 
		dataset - [ num_samples x dim ] 2D numpy matrix containing data samples
		n_neighbours - Integer, specifying number of nearest neighbours to consider for checking violations (considers all pairs by default)
		title - string, title to be printed along with results
		verbose - bool, prints the violation report when set to True
		stat_distance_metric - statistical distance to be used (refer to Args.py)
		euclidean_distance metric - euclidean distance to be used (refer to Args.py)
		normalise_stat - True if statistical distance should be normalised to [0, 1]
		normalise_euclidean - True if euclidean_distance should be normalised to [0, 1]
		fairness_type - 0 <- metric fairness 1 <- local fairness
	Output:
		percentage_violations - integer, the percentage of total constraints that get violated. 
	"""

	num_samples = dataset.shape[0]

	# Consider all pairs if n_neighbours = 0
	if(n_neighbours == 0):
		n_neighbours = num_samples - 1

	# Build ball_tree for kNN queries
	knn = NearestNeighbors(n_neighbors = n_neighbours + 1, algorithm = "ball_tree").fit(dataset)	# One more than n_neighbours as query point itself is a neighbour which is ignored
	_, neighbours = knn.kneighbors(dataset)
	neighbours = neighbours[:, 1:]			# Remove the first nearest neighbour which is itself

	# Create selection mask - selection[i][j] is true if j is one of top 'n_neighbours' neighbours of i.
	selection = np.zeros([num_samples, num_samples]).astype(bool)
	row_id = np.asarray([i for i in range(num_samples) for j in range(n_neighbours)]).reshape(-1)
	selection[row_id, neighbours.reshape(-1)] = True

	# Compute pairwise statistical distance
	stat_distance = FindStatisticalDistance(distribution, stat_distance_metric, normalise_stat)
	stat_distance = stat_distance[selection].reshape(num_samples, n_neighbours)

	# Compute pairwise euclidean distance
	euclidean_distance = FindEuclideanDistance(dataset, euclidean_distance_metric, normalise_euclidean)
	euclidean_distance = euclidean_distance[selection].reshape(num_samples, n_neighbours)

	if(fairness_type == 1):
		euclidean_distance = (euclidean_distance) / (np.max(euclidean_distance, axis = 1)).reshape(-1, 1)

	total_pairs, violating_pairs = CountViolations(euclidean_distance, stat_distance)
	percentage_violations = (violating_pairs / (total_pairs + MU_EPS)) * 100

	if(verbose):
		print("RESULTS - " + title)
		print("Nearest neighbours 			: {n_neighbours}".format(
				n_neighbours = n_neighbours
			))
		print("Total Constraints 			: {total_pairs}".format(
				total_pairs = total_pairs
			))
		print("Violating Constraints 			: {violating_pairs}".format(
				violating_pairs = violating_pairs
			))
		print("Percentage Violations 			: {percentage_violations}".format(
				percentage_violations = round(percentage_violations, 2)
			))
		print()

	return percentage_violations 

def GroupwisePercentageViolations(distribution, dataset, groups, n_neighbours = 0, title = "", verbose = False,
			stat_distance_metric = 0, euclidean_distance_metric = 0, 
			normalise_stat = True, normalise_euclidean = True, fairness_type = 0
		):

	"""
	Given the dataset and distribution over the points, this function computers number of violations of Lipschitz constraint
	Input:
		distribution - [ num_samples x num_clusters ] 2D numpy matrix containing distribution over cluster centres for each data sample 
		dataset - [ num_samples x dim ] 2D numpy matrix containing data samples
		groups - [ num_samples x num_protected_features ] 2D matrix containing a list of groups that the ith sample belongs to 
		n_neighbours - No of nearest neighbours to consider for indv. fairness violation computation. Note that two points will be considered only if they belong to some common protected group 
		title - string, title to be printed along with results
		verbose - bool, prints the violation report when set to True
		stat_distance_metric - statistical distance to be used (refer to Args.py)
		euclidean_distance metric - euclidean distance to be used (refer to Args.py)
		normalise_stat - True if statistical distance should be normalised to [0, 1]
		normalise_euclidean - True if euclidean_distance should be normalised to [0, 1]
		fairness_type - 0 <- metric fairness 1 <- local fairness
	Output:
		percentage_violations - integer, the percentage of good constraint pairs getting violated
								A constraint pair i, j is called good if i and j are a part of some same protected group 
	"""

	if(len(groups.shape) == 1):
		groups.reshape(len(groups), 1)

	num_samples = dataset.shape[0]
	num_protected_features = groups.shape[1]

	# Consider all pairs if n_neighbours = 0
	if(n_neighbours == 0):
		n_neighbours = num_samples - 1

	# Build ball_tree for kNN queries
	knn = NearestNeighbors(n_neighbors = n_neighbours + 1, algorithm = "ball_tree").fit(dataset)	# One more than n_neighbours as query point itself is a neighbour which is ignored
	_, neighbours = knn.kneighbors(dataset)
	neighbours = neighbours[:, 1:]			# Remove the first nearest neighbour which is itself

	# Create selection mask - _selection[i][j] is true if j is one of top 'n_neighbours' neighbours of i.
	_selection = np.zeros([num_samples, num_samples]).astype(bool)
	row_id = np.asarray([i for i in range(num_samples) for j in range(n_neighbours)]).reshape(-1)
	_selection[row_id, neighbours.reshape(-1)] = True

	# Compute pairwise statistical distance
	stat_distance = FindStatisticalDistance(distribution, stat_distance_metric, normalise_stat)

	# Compute pairwise euclidean distance
	euclidean_distance = FindEuclideanDistance(dataset, euclidean_distance_metric, normalise_euclidean)

	if(fairness_type == 1):

		threshold = np.copy(euclidean_distance)
		threshold = threshold[_selection].reshape(num_samples, n_neighbours)

		euclidean_distance = (euclidean_distance) / (np.max(threshold, axis = 1)).reshape(-1, 1)
		euclidean_distance[np.logical_not(_selection)] = 1

	total_pairs = 0
	violating_pairs = 0
	selection = np.zeros([num_samples, num_samples]).astype(bool)

	# Iterate over protected feautres
	for i in range(num_protected_features):
		protected_groups = list(set(groups[:, i]))
		# Iterate over protected groups
		for group in protected_groups:
			mask = np.where(groups[:, i] == group)[0]
			mask = [np.tile(mask, len(mask)), np.repeat(mask, len(mask))]
			selection[mask[0], mask[1]] = True 	# selection i, j = True if and only if i, j belong to some common protected group

	for i in range(num_samples):
		selection[i][i] = False

	# Combine both the selection masks to enforce constraints over the intersection.
	selection = selection & _selection

	# Compute where stat_distance > euclidean_distance only for selected pairs based on selection mask computed in the previous step
	violations = (stat_distance - MU_EPS > euclidean_distance) & (selection)
	violating_pairs = violations.sum()
	total_pairs = selection.sum()

	percentage_violations = (violating_pairs / total_pairs) * 100

	# Report results
	if(verbose):
		print("RESULTS - " + title)
		print("Total Constraints 			: {total_pairs}".format(
				total_pairs = total_pairs
			))
		print("Violating Constraints 			: {violating_pairs}".format(
				violating_pairs = violating_pairs
			))
		print("Percentage Violations 			: {percentage_violations}".format(
				percentage_violations = round(percentage_violations,2)
			))
		print()

	return percentage_violations

def Bias(distribution, dataset, groups, verbose = False):

	"""
	This function computes the maximum bias (statistical violations) and the corresponding earthmover distance 

	Input:
		distribution - [ num_samples x num_clusters ] 2D numpy matrix containing distribution over cluster centres for each data sample 
		dataset 	 - [ num_samples x dim ] 2D numpy matrix containing data samples
		groups  	 - [ num_samples x num_protected_features ] 2D matrix containing a list of groups that the ith sample belongs to 
		verbose  	 - bool, prints the violation report when set to True
	Output:
		bias 	  	 - Returns a single float, the amount of bias in the clustering solution 
	"""

	# Convert 1D array to a 2D array
	if(len(groups.shape) == 1):
		groups = groups.reshape(-1, 1)

	distance = FindEuclideanDistance(dataset, 0, True)
	num_protected_features = groups.shape[1]
	num_samples = groups.shape[0]

	bias = []
	bias_without_mod = []
	earthmover = []

	total_mass_on_cluster = distribution.sum(axis = 0) 
	
	max_bias = (-1, -1) 		# bias, earthmover

	# Compute bias
	for i in range(num_protected_features):
		
		protected_groups = list(set(groups[:, i]))
		protected_groups.sort()
		
		bias.append(np.zeros([np.max(protected_groups) + 1]))
		bias_without_mod.append(np.zeros([np.max(protected_groups) + 1]))
		earthmover.append(np.zeros([np.max(protected_groups) + 1]))
		
		for g in protected_groups:
			

			num_pts_in_group = (groups[:, i] == g).sum()
			if(num_pts_in_group == 0): 		# Don't process empty groups
				continue
			# p_r, as mentioned in the paper
			frac_pts_in_group = num_pts_in_group / num_samples
			# mass contributed by members within the group
			group_distribution = distribution[groups[:, i] == g].sum(axis = 0) 
			# Compute bias
			bias[i][g] = np.max(np.abs((group_distribution / num_pts_in_group) - (total_mass_on_cluster / num_samples)))
			bias_without_mod[i][g] = np.max((group_distribution / num_pts_in_group) - (total_mass_on_cluster / num_samples))

			# Let S be the set of points within a group. S_distribution = 1 / |S| for all points in S
			S_distribution = np.zeros([num_samples])
			S_distribution[groups[:, i] == g] =  1 / num_pts_in_group
			# Let T be the set of all points. T_distribution = 1 / |T| for all points in T
			T_distribution = np.zeros([num_samples])
			T_distribution[:] = 1 / num_samples
			earthmover[i][g] = emd(S_distribution, T_distribution, distance) 

			if(bias_without_mod[i][g] > max_bias[0]):
				max_bias = (bias_without_mod[i][g], earthmover[i][g])	

	# Return maximum bias and corresponding earthmover distance over all the groups
	return max_bias

if(__name__ == "__main__"):

	pass
