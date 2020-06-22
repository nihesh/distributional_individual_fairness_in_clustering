# File		: gf.py

"""
This module implements Group fairness (GF) algorithm proposed in "Fair Algorithms for Clustering" - Bera et al
"""

import numpy as np
from kmeans import LoadPickledData, FindClusters
from cplex import Cplex
from utilities.stat_lp_tools import *
from utilities.utils import labelToOneHot, GroupwisePercentageViolations, PercentageViolations, Bias, ResetWorkspace
from utilities.result import Result
from utilities.cost_metric import KMeansCost
import os
import sys
import time

DATASET = "bank" # <bank/adult/creditcard/census1990/diabetes> is a folder in ./data/processed containing subsampled batches of the dataset

# Read name of dataset from command-line, if it's available
if(len(sys.argv) > 1):
	DATASET = sys.argv[1]
	
# Script arguments
PICKLED_DATASET = "./data/processed/" + DATASET		
NUM_CLUSTERS = [6, 2, 4, 8, 10]
NUM_FILES = 1										# Maximum number of subsampled files to run the algorithm on (atmost the number of files in the folder mentioned in PICKLED_DATASET)
NUM_SAMPLES = 1000 									# Minimum of NUM_SAMPLES and number of available points will be used
NUM_NEAREST_NEIGHBOURS = 250						# Number of nearest neighbours to evaluate individual fairness constraint violations 
DELTA = 0.2	 										# Range = [0, 1) - Usually very small
QUIET = False										# Doesn't print simulation results to stdout when set to True
FAIRNESS_TYPE = 1
if(len(sys.argv) > 2):
	FAIRNESS_TYPE = int(sys.argv[2])

DUMP_PATH = "./Simulation_Dump/fairness_type_" + str(FAIRNESS_TYPE) + "/" + DATASET + "/"

def PrintSimulationParameters(pickled_dataset, num_clusters, result):

	global DELTA, NUM_SAMPLES, QUIET

	result.AddSimulationInfo("Dataset used", pickled_dataset)
	result.AddSimulationInfo("Number of samples", NUM_SAMPLES)
	result.AddSimulationInfo("Num clusters", num_clusters)
	result.AddSimulationInfo("Delta", DELTA)
	result.AddSimulationInfo("Statistical Distance", "TV Distance")
	result.AddSimulationInfo("Euclidean Distance", "L2 Distance")
	result.AddSimulationInfo("Normalise Stat Distance?", "No")
	result.AddSimulationInfo("Normalise Euc Distance?", "Yes")

	if(QUIET):
		return

	print(result)

def fraction_of_protected_group(groups, delta):

	"""
	Computes the fraction of data points belonging to a protected group and returns the corresponding alpha and beta values
	computed as following
	alpha = (1 / (1 - delta)) * (size of group / num_samples), beta = (1 - delta) * (size of group / num_samples)

	Input:
		groups 	- [ num_samples x num_protected_features ] - 2D numpy array where the ith column partitions the dataset based on an attribute 
		delta 	- scaling constant
	Output:
		Returns Alpha[i][j], Beta[i][j] - The constants mentioned in statistical fairness paper for ith protected_feature and jth protected_group
	"""

	# Convert 1D array to a 2D array
	if(len(groups.shape) == 1):
		groups = groups.reshape(-1, 1)

	num_protected_features = groups.shape[1]
	num_samples = groups.shape[0]

	alpha = {}
	beta = {}

	# Compute Alpha and Beta
	for i in range(num_protected_features):
		protected_groups = list(set(groups[:, i]))
		protected_groups.sort()
		alpha[i] = np.zeros([np.max(protected_groups) + 1])
		beta[i] = np.zeros([np.max(protected_groups) + 1])
		for g in protected_groups:
			alpha[i][g] = (groups[:, i] == g).sum() / num_samples
			beta[i][g] = (groups[:, i] == g).sum() / num_samples

	# Scale Alpha and Beta by delta
	for i in range(num_protected_features):
		alpha[i] = alpha[i] * (1 / (1 - delta))
		beta[i] = beta[i] * (1 - delta)

	return alpha, beta

def stat_clustering(dataset, centres, groups, delta):

	"""
	Implements statistically fair clustering algorithm and returns the results of lp solution
	Input:
		dataset 	- [ num_samples x dim ] 2D numpy array containing data
		centres 	- [ num_clusters x dim ] 2D numpy array containing centres obtained using some clustering algorithm
		groups 		- [ num_clusters x num_categories ] 2D numpy array partitioning the entire dataset into different protected groups 
					  (2D because a single point can belong to multiple protected groups)
		delta 		- alpha = (1 / (1 - delta)) * fraction of protected group, beta = (1 - delta) * fraction of protected group
	Output:
		Result - dictionary containing
					- status
					- success
					- objective 	- solution cost
					- assignment 	- values assigned to lp variabless
	"""

	# Step 1: 	Create an instance of Cplex 
	problem = Cplex()

	# Step 2: 	Declare that this is a minimization problem
	problem.objective.set_sense(problem.objective.sense.minimize)

	# Step 3.   Declare and  add variables to the model. The function
	#           prepare_to_add_variables (dataset, centres) prepares all the
	#           required information for this stage.
	#
	#    objective: a list of coefficients (float) in the linear objective function
	#    lower bound: a list of floats containing the lower bounds for each variable
	#    upper bound: a list of floats containing the upper bounds for each variable
	#    variable_names: a list of strings that contains the name of the variables

	objective, lower_bound, upper_bound, variable_names, P = prepare_to_add_variables(dataset, centres)
	problem.variables.add(
			obj = objective,
			lb = lower_bound,
			ub = upper_bound,
			names = variable_names
		)

	# Step 4.   Declare and add constraints to the model.
	#           There are few ways of adding constraints: row wise, col wise and non-zero entry wise.
	#           Assume the constraint matrix is A. We add the constraints non-zero entry wise.
	#           The function prepare_to_add_constraints(dataset, centres)
	#           prepares the required data for this step.
	#
	#  coefficients: Three tuple containing the row number, column number and the value of the constraint matrix
	#  senses: a list of strings that identifies whether the corresponding constraint is
	#          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
	#  rhs: a list of floats corresponding to the rhs of the constraints.
	#  constraint_names: a list of string corresponding to the name of the constraint

	alpha, beta = fraction_of_protected_group(groups, delta)
	rhs, senses, row_names, coefficients = prepare_to_add_constraints(dataset, centres, P, groups, alpha, beta)
	problem.linear_constraints.add(
			rhs = rhs,
			senses = senses,
			names = row_names
		)
	problem.linear_constraints.set_coefficients(coefficients)

	# Step 5.	Solve the problem
	problem.solve()

	result = {
		"status": problem.solution.get_status(),
		"success": problem.solution.get_status_string(),
		"objective": problem.solution.get_objective_value(),
		"assignment": problem.solution.get_values(),
	}

	return result

def StatisticalFairness(pickled_dataset, num_clusters, result, fairness_type):

	"""
	Main function for Statistical Fairness Algorithm
	"""

	global NUM_SAMPLES, DELTA, NUM_NEAREST_NEIGHBOURS

	dataset, groups = LoadPickledData(pickled_dataset)

	# Subsample the dataset uniformly at random, assuming the rows in pkl file are randomly shuffled already
	dataset = dataset[: NUM_SAMPLES]
	groups = groups[: NUM_SAMPLES]

	NUM_NEAREST_NEIGHBOURS = min(NUM_NEAREST_NEIGHBOURS, dataset.shape[0] - 1)

	# Convert groups to 2D in case it's 1D
	if(len(groups.shape) == 1):
		groups = groups.reshape(-1, 1)

	# Vanilla clustering
	_, centres, labels = FindClusters(dataset, num_clusters, 0, 1)

	# KMeans hard assignment cost
	hard_distribution = labelToOneHot(labels)
	hard_cost = KMeansCost(dataset, centres, hard_distribution)
	num_samples = len(dataset)

	# Uniform assignment cost - worst case
	uniform_distribution = np.zeros([num_samples, num_clusters]).astype(float)
	uniform_distribution[:,:] = 1 / num_clusters
	uniform_cost = KMeansCost(dataset, centres, uniform_distribution)

	# LP for statistical fairness clustering
	output = stat_clustering(dataset, centres, groups, DELTA)

	fair_cost = output["objective"] / num_samples

	# Print Simulation Environment Parameters
	PrintSimulationParameters(pickled_dataset, num_clusters, result)

	result.AddCost("Hard KMeans", round(hard_cost, 2))
	result.AddCost("Statistical fairness", round(fair_cost, 2))
	result.AddCost("Uniform assignment", round(uniform_cost, 2))
	result.AddCost("Percentage decrease (hard vs alg)", round(((fair_cost - hard_cost) / fair_cost) * 100, 2))
	result.AddCost("Percentage increase (uniform vs alg)", round(((uniform_cost - fair_cost) / fair_cost) * 100, 2))

	# Report the number of constraint violations
	optimal_assignment = output["assignment"][:num_samples * num_clusters]						# Extract P_i_j values alone and discard C_i_j_k
	distribution = np.asarray(optimal_assignment).reshape(num_samples, num_clusters)			# Convert matrix to 2D representation
	group_violations = GroupwisePercentageViolations(distribution, dataset, groups, NUM_NEAREST_NEIGHBOURS,
			title = "Constraint Violations within protected groups",
			verbose = not QUIET,
			stat_distance_metric = 0,			# 0 = TV distance - LP doesn't support D_inf yet 
			euclidean_distance_metric = 0, 		# 0 = L2 distance
			normalise_stat = False,				# TV distance is already normalised 
			normalise_euclidean = True,			# Euclidean distance has to be normalised
			fairness_type = fairness_type
		)

	percentage_violations = PercentageViolations(distribution, dataset, NUM_NEAREST_NEIGHBOURS,
			title = "Constraint Violations over all pairs",
			verbose = not QUIET,
			stat_distance_metric = 0,			# 0 = TV distance - LP doesn't support D_inf yet 
			euclidean_distance_metric = 0, 		# 0 = L2 distance
			normalise_stat = False,				# TV distance is already normalised 
			normalise_euclidean = True,			# Euclidean distance has to be normalised
			fairness_type = fairness_type
		)

	# Compute statistical bias
	stat_bias, earthmover = Bias(distribution, dataset, groups, verbose = not QUIET)

	result.AddCost("PercentageViolations", round(percentage_violations, 2))
	result.AddCost("GroupwiseViolations", round(group_violations, 2))
	result.AddCost("Statistical Bias", round(stat_bias, 2))
	result.AddCost("Earthmover Distance", round(earthmover, 2))
	result.AddSolution(centres, distribution)
	
	if(not QUIET):
		# Report solution cost
		print(result.GetCost())

if(__name__ == "__main__"):

	ResetWorkspace()

	files = os.listdir(PICKLED_DATASET)
	files.sort(key=lambda item: (len(item), item))
	files = files[:NUM_FILES]
	target = os.path.join(DUMP_PATH, "statistical_fairness_" + str(NUM_SAMPLES))	

	# Remove existing files from previous run
	os.system("rm -rf " + target)

	for num_clusters in NUM_CLUSTERS:				# Compute result for varying cluster size
		for file in files:							# Average over random subsamples of the entire dataset

			if(FAIRNESS_TYPE == 1):

				NUM_NEAREST_NEIGHBOURS = NUM_SAMPLES // num_clusters

			print("\nSolving " + file + " for num_clusters = " + str(num_clusters) + "...")

			data_file = os.path.join(PICKLED_DATASET, file)					# Path to the current batch of data
			result = Result("Statistical Fairness", num_clusters, file, target)		# Instance of result which will be dumped
			start = time.time()
			StatisticalFairness(data_file, num_clusters, result, FAIRNESS_TYPE)
			end = time.time()

			result.addRunningTime(end - start)
			result.dump()													# Dump the result

	






