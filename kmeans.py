# File	: kmeans.py

"""
This module implements SKM (soft k-means)
"""

# Dependencies
import time
from sklearn.cluster import KMeans
import pickle
from utilities.utils import *
from utilities.cost_metric import *
from utilities.result import Result
import sys

import os

# Script Arguments

DATASET = "bank" # <bank/adult/creditcard/census1990/diabetes> is a folder in ./data/processed containing subsampled batches of the dataset
# Read name of dataset from command-line, if it's available
if(len(sys.argv) > 1):
	DATASET = sys.argv[1]

PICKLED_DATASET = "./data/processed/" + DATASET		
NUM_CLUSTERS = [10, 2, 4, 8, 6]
RANDOM_STATE = 0	# Seed value to reproduce results
NUM_SAMPLES = 1000 	# Minimum of NUM_SAMPLES and number of available points will be used
NUM_FILES = 1		# Number of files to average the results over
QUIET = False		# If QUIET, output will not be printed to stdout
FAIRNESS_TYPE = 1
if(len(sys.argv) > 2):
	FAIRNESS_TYPE = int(sys.argv[2])
	
DUMP_PATH = "./Simulation_Dump/fairness_type_" + str(FAIRNESS_TYPE) + "/" + DATASET + "/"

NUM_NEAREST_NEIGHBOURS = 250	# Number of nearest neighbours for printing verbose violation results

# Graph parameters - x-axis min, max and step size
NUM_NEIGHBOURS_MIN = 5			# Inclusive
NUM_NEIGHBOURS_MAX = 1000		# Not inclusive
NUM_NEIGHBOURS_STEP = 50		# Step size

DISTRIBUTION = 2	
"""
DISTRIBUTION specifies the kind of distribution to be used over the clustered data points
Values:
	0 - nearest neighbour one hot encoded distribution
	1 - Inverse L2 distance from centres, normalised by sum
	2 - Negative L2 distance from centres, normalised by softmax with scaling (tune temperature parameter to modify cost gain)
"""

TEMPERATURE = 4				# Temperature value for printing verbose violation and cost increase results

# Graph parameters
TEMPERATURE_MIN = 0			# Temperature parameter is associated with DISTRIBUTION = 2 only - Min temperature in the plot
TEMPERATURE_MAX = 6 		# Temperature parameter is associated with DISTRIBUTION = 2 only - Max temperature in the plot
TEMPERATURE_STEP = 0.03	 	# Temperature step size

STAT_DISTANCE_METRIC = 0
NORMALISE_STAT = False 		# Works only for L Infinity distance - Normalises the distance to [0, 1]
"""
STAT_DISTANCE_METRIC specifies the type of statistical distance used to compare two probability distributions
Values:
	0 - Total variation distance
	1 - L Infinity distance
"""

EUCLIDEAN_DISTANCE_METRIC = 0
NORMALISE_EUCLIDEAN = True	# Normalises euclidean distance to [0, 1]
"""
STAT_DISTANCE_METRIC specifies the type of statistical distance used to compare two probability distributions
Values:
	0 - L2 Distance
"""

EPS = 1e-9 	# Constant added to distance to prevent divide by zero error

# End of arguments

def PrintSimulationParameters(num_clusters, pickled_dataset, result):

	global NUM_SAMPLES, NUM_NEAREST_NEIGHBOURS, DISTRIBUTION, QUIET
	global NORMALISE_EUCLIDEAN, NORMALISE_STAT, EUCLIDEAN_DISTANCE_METRIC, STAT_DISTANCE_METRIC

	yes_or_no = {0: "No", 1: "Yes"}

	# Code to string mapping for verbose printing
	euc_dist_metric = {
		0: "L2 Distance",
	}

	stat_dist_metric = {
		0: "TV Distance",
		1: "D_Inf Distance"
	}

	cluster_method = {
		0: "Hard Clustering",
		1: "Inverse L2 Distance normalised by sum",
		2: "Negative L2 Distance normalised by softmax"
	}

	euc_dist_metric = euc_dist_metric[EUCLIDEAN_DISTANCE_METRIC]
	stat_dist_metric = stat_dist_metric[STAT_DISTANCE_METRIC]
	cluster_method = cluster_method[DISTRIBUTION]

	# Add simulation environment details to the result object 
	result.AddSimulationInfo("Dataset used", pickled_dataset)
	result.AddSimulationInfo("Number of clusters", num_clusters)
	result.AddSimulationInfo("Num samples", NUM_SAMPLES)
	result.AddSimulationInfo("Num nearest neighbours", NUM_NEAREST_NEIGHBOURS)
	result.AddSimulationInfo("Soft Clustering Method", cluster_method)
	result.AddSimulationInfo("Euclidean distance metric", euc_dist_metric)
	result.AddSimulationInfo("Normalise Euc distance?", yes_or_no[NORMALISE_EUCLIDEAN])
	result.AddSimulationInfo("Statistical distance metric", stat_dist_metric)
	result.AddSimulationInfo("Normalise Stat distance?", yes_or_no[NORMALISE_STAT] if(STAT_DISTANCE_METRIC == 1) else "No")

	if(QUIET):
		return result

	# Printing out the simulation parameters
	print("\nVanilla kmeans with softmax implementation\n")

	print(result)

	return result

def LoadPickledData(src):

	"""
	Input: 
		src - Source path to the pickled 2D numpy array
	Output:
		Numpy array containing data - [ Num_pts x Dim ] matrix
	"""

	file = open(src, "rb")
	dataset = pickle.load(file)
	file.close()

	return dataset

def FindClusters(dataset, num_clusters, distribution, temperature):

	"""
	Input:
		dataset - [ num_samples x dim ] matrix containing data to be clustered
		num_clusters - number of cluster centres
		distribution - Type of distribution to be returned - eg. softmax, hard clustering, etc (codes have been mentioned in script arguments)
	Output:
		(distribution, cluster_centres)
		distribution - [ num_samples x num_clusters ] matrix containing distribution of the ith point
		cluster_centres - [ num_clusters x dim ] matrix containing the label/class centres
		labels - labels to which points are assigned to, in hard clustering. 
	"""

	global RANDOM_STATE

	kmeans = KMeans(n_clusters = num_clusters, init = "k-means++", random_state = RANDOM_STATE)
	kmeans = kmeans.fit(dataset)
	
	distribution = computeDistribution(dataset, kmeans, distribution, temperature)

	# Build ball_tree for NN queries
	knn = NearestNeighbors(n_neighbors = 1, algorithm = "ball_tree").fit(dataset)	
	
	# Approximate the obtained cluster centres with the nearest neighbour in the dataset. 
	_, kmeans.cluster_centers_ = knn.kneighbors(kmeans.cluster_centers_)
	kmeans.cluster_centers_ = dataset[kmeans.cluster_centers_.reshape(-1)]

	return distribution, kmeans.cluster_centers_, kmeans.labels_

def ComputeRelativeGain(dataset, centres, distribution, labels):

	"""
	Computes the cluster cost for hard assignment and the chosen assignment technique and computes the percentage decrease in cost of hard assignment wrt ALG
	We ideally want the relative decrease to be less, i.e., soft solution is closer to hard solution
	Input:
		dataset - [ num_samples x dim ] 2D numpy matrix containing data samples
		centres - [ num_clusters x dim ] 2D numpy matrix containing cluster centres
		distribution - [ num_samples x num_clusters ] 2D numpy matrix containing distribution over cluster centres for each data sample 
		labels - [ num_samples ] 1D numpy vector denoting the hard clusters assigned to data samples
 	Output:
		(hard cost, proposed cost, gain)
		hard cost - solution cost of 0-1 distribution kmeans
		proposed cost - solution cost of the used distribution
		decrease - percentage decrease in hard cost wrt ALG 
	"""

	hard_distribution = labelToOneHot(labels)
	hard_cost = KMeansCost(dataset, centres, hard_distribution)
	proposed_cost = KMeansCost(dataset, centres, distribution)

	decrease = (proposed_cost - hard_cost) / proposed_cost
	decrease *= 100			# Percentage computation

	return hard_cost, proposed_cost, decrease 

def SoftKMeans(pickled_dataset, num_clusters, result, fairness_type):

	"""
	Main function for Soft KMeans algorithm
	"""

	global NUM_SAMPLES, DISTRIBUTION, TEMPERATURE, NUM_NEAREST_NEIGHBOURS, STAT_DISTANCE_METRIC, \
				EUCLIDEAN_DISTANCE_METRIC, NORMALISE_STAT, NORMALISE_EUCLIDEAN, TEMPERATURE_MIN, TEMPERATURE_MAX, \
				TEMPERATURE_STEP, TEMPERATURE_CURVE_SAVE_PATH, NUM_NEIGHBOURS_MIN, NUM_NEIGHBOURS_MAX, NUM_NEIGHBOURS_STEP, \
				NUM_NEIGHBOURS_CURVE_SAVE_PATH, QUIET


	dataset, _ = LoadPickledData(pickled_dataset)

	# Subsample from the dataset - Happens uniformly at random if the records in pkl file are randomly shuffled
	dataset = dataset[: NUM_SAMPLES]

	num_samples = dataset.shape[0]
	distribution, centres, labels = FindClusters(dataset, num_clusters, DISTRIBUTION, TEMPERATURE)
	hard_cost, proposed_cost, cost_decrease = ComputeRelativeGain(dataset, centres, distribution, labels)

	# Uniform assignment cost - worst case
	uniform_distribution = np.zeros([num_samples, num_clusters]).astype(float)
	uniform_distribution[:,:] = 1 / num_clusters
	uniform_cost = KMeansCost(dataset, centres, uniform_distribution)
	cost_increase = ((uniform_cost - proposed_cost) / proposed_cost) * 100

	# Print simulation environment parameters
	PrintSimulationParameters(num_clusters, pickled_dataset, result)

	result.AddCost("Hard K-Means", round(hard_cost, 2))
	result.AddCost("Soft K-Means", round(proposed_cost, 2))
	result.AddCost("Uniform", round(uniform_cost, 2))
	result.AddCost("Percentage decrease (hard wrt alg)", round(cost_decrease, 2))
	result.AddCost("Percentage increase (uniform wrt alg)", round(cost_increase, 2))

	# Print the number of overall violations 
	percentage_violations = PercentageViolations(distribution, dataset, NUM_NEAREST_NEIGHBOURS,
			title = "Constraint Violations", verbose = not QUIET,
			stat_distance_metric = STAT_DISTANCE_METRIC, 
			euclidean_distance_metric = EUCLIDEAN_DISTANCE_METRIC, 
			normalise_stat = NORMALISE_STAT, 
			normalise_euclidean = NORMALISE_EUCLIDEAN,
			fairness_type = FAIRNESS_TYPE
		)
	result.AddCost("PercentageViolations 			:", round(percentage_violations, 2))
	result.AddSolution(centres, distribution)
	
	if(not QUIET):
		# Cost is defined as the expected distance to the cluster centre for a single data sample
		print(result.GetCost())

	if(not QUIET):
		# Count the number of points assigned to each cluster
		print("Statistics - Num of samples in a cluster")
		for cluster_id in range(num_clusters):

			print("Cluster {cluster_id}				: {num_samples}".format(
					cluster_id = cluster_id,
					num_samples = (labels == cluster_id).sum()
				))
		print()

	# Save plot of k means cost and percentage violations vs stifness parameter

	stiffness_axis = []
	cost_decrease_axis = []
	violations_axis = []

	# Iterate over different temperature values and generate data to create plot
	print("Generating Cost vs Violations Trade-off...")
	temperature = TEMPERATURE_MIN
	while(temperature < TEMPERATURE_MAX + EPS):

		distribution, centres, labels = FindClusters(dataset, num_clusters, DISTRIBUTION, temperature)

		perc_violations = PercentageViolations(distribution, dataset, NUM_NEAREST_NEIGHBOURS,
									verbose = False,
									stat_distance_metric = STAT_DISTANCE_METRIC, 
									euclidean_distance_metric = EUCLIDEAN_DISTANCE_METRIC, 
									normalise_stat = NORMALISE_STAT, 
									normalise_euclidean = NORMALISE_EUCLIDEAN,
									fairness_type = FAIRNESS_TYPE
								)
		hard_cost, proposed_cost, cost_decrease = ComputeRelativeGain(dataset, centres, distribution, labels)

		stiffness_axis.append(temperature)
		cost_decrease_axis.append(cost_decrease)
		violations_axis.append(perc_violations)

		temperature += TEMPERATURE_STEP

	print("Generated Cost vs Violations Trade-off...")	
	result.AddSecondaryAxisPlot([
			stiffness_axis, cost_decrease_axis, violations_axis, 
			"Stiffness parameter",				
			"K Means percentage cost decrease (HKM w.r.t. SKM)",
			"Percentage of constraint violations",
			"Cost vs Violations Trade-off"
		])

	# Compute percentage violations for different values of num_nearest_neighbours
	print("\nGenerating Num neighbours vs Percentage violations...")

	n_neighbours = NUM_NEIGHBOURS_MIN
	n_neighbours_axis = []
	violations_axis = []
	while(n_neighbours < NUM_NEIGHBOURS_MAX):

		# Compute violations for different values of n_neighbours
		perc_violations = PercentageViolations(distribution, dataset, n_neighbours,
									verbose = False,
									stat_distance_metric = STAT_DISTANCE_METRIC, 
									euclidean_distance_metric = EUCLIDEAN_DISTANCE_METRIC, 
									normalise_stat = NORMALISE_STAT, 
									normalise_euclidean = NORMALISE_EUCLIDEAN,
									fairness_type = FAIRNESS_TYPE
								)

		# Populate the data list
		n_neighbours_axis.append(n_neighbours)
		violations_axis.append(perc_violations)

		n_neighbours += NUM_NEIGHBOURS_STEP

	# Store the generated data list
	result.AddSingleAxisPlot([
			n_neighbours_axis, violations_axis, 
			"Number of nearest neighbours considered",
			"Percentage of constraints violated",
			"Num neighbours vs Percentage violations"
		])

	print("Generated Num neighbours vs Percentage violations...")

if(__name__ == "__main__"):

	ResetWorkspace()

	files = os.listdir(PICKLED_DATASET)
	files.sort(key=lambda item: (len(item), item))
	files = files[:NUM_FILES]
	target = os.path.join(DUMP_PATH, "soft_kmeans_" + str(NUM_SAMPLES))	

	# Remove existing files for soft_kmeans 
	os.system("rm -rf " + target)

	for num_clusters in NUM_CLUSTERS:				# Compute results for varying cluster size
		for file in files:							# Average over random subsamples of the entire dataset

			if(FAIRNESS_TYPE == 1):

				NUM_NEAREST_NEIGHBOURS = NUM_SAMPLES // num_clusters

			print("\nSolving " + file + " for num_clusters = " + str(num_clusters) + "...")

			data_file = os.path.join(PICKLED_DATASET, file)					# Path to the current batch of data
			result = Result("Soft_KMeans", num_clusters, file, target)		# Instance of result which will be dumped
			start = time.time()
			SoftKMeans(data_file, num_clusters, result, FAIRNESS_TYPE)
			end = time.time()

			result.addRunningTime(end - start)
			result.dump()													# Dump the result
