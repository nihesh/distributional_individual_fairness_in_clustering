#File	: inference.py

from utilities.result import Result
import pickle
import os
import numpy as np
import matplotlib
import matplotlib.font_manager
from matplotlib import pyplot as plt
from kmeans import LoadPickledData
from sklearn.neighbors import NearestNeighbors
import utilities.utils as utils
import sys

# Definitions
COLOR_CODE = {
	"light_red" 		: "#ff474c",
	"light_purple" 		: "#bf77f6",
	"light_blue"		: "#0485d1",
	"dark_green" 		: "#40a368",
	"dark_grey" 		: "#363737"
}
SHORT_ALIAS = {
	"soft_kmeans" 		   					: "SKM",
	"individual_fairness" 					: "ALG-IF",
	"statistical_fairness"					: "GF",
	"statistical_individual_fairness" 		: "ALG-CF",
	"opt_individual_fairness" 			 	: "OPT-IF",
	"opt_statistical_individual_fairness" 	: "OPT-CF"
}
LONG_ALIAS = {
	"soft_kmeans" 		   				: "Soft K-Means",
	"individual_fairness" 				: "Individual Fairness",
	"statistical_fairness" 				: "Group Fairness",
	"statistical_individual_fairness"	: "Combined Fairness"	
}

# Script parameters

FAIRNESS_TYPE = 0
if(len(sys.argv) > 1):
	FAIRNESS_TYPE = int(sys.argv[1])

ALG_DATASET_SIZE = 1000				# Dataset size used for running the approximation algorithm (usually 1000)
OPT_DATASET_SIZE = 80 				# Dataset size used for running the opt algorithm (usually 80) 
DATASET = ["bank", "adult", "creditcard", "census1990", "diabetes"] 	# <bank/adult/creditcard/census/diabetes>		

PICKLED_DATASET = "./data/processed/"
DUMP_PATH = "./Simulation_Dump/fairness_type_" + str(FAIRNESS_TYPE) 
INFERENCE_SAVE_PATH = "./Inferences/fairness_type_" + str(FAIRNESS_TYPE) 
TIMING_RESULT_PATH = "./Simulation_Dump/fairness_type_0/creditcard/variable_size/4_clusters"

# Used for computing IF violations after realizing a sample from the distribution
NUM_NEIGHBOURS = 250				# Ideally, magnitude of the hyperparameter is the same as the one used while solving the LP

# Graph Attributes
LINE_STYLE = ["solid", "dashed", "dashdot", "dotted", (0, (5, 1))]
COLOR 	   = [COLOR_CODE["light_red"], COLOR_CODE["light_blue"], COLOR_CODE["dark_green"], COLOR_CODE["dark_grey"], COLOR_CODE["light_purple"]]
MARKER 	   = ["o", "s", "d", "v", "p"]

# Appearance Parameters
LABEL_FONT_SIZE = 22
AXIS_FONT_SIZE = 15

def computeAverage(path, dataset, dataset_src):

	"""
	Input:
		path 		- Path to .result files
		dataset 	- Type of dataset
		dataset_src - Source path to processed data file
	Output:
		returns a result object containing averaged values over the result files in path.
	"""


	result_files = os.listdir(path)

	output = []

	# Read all result pkl files and insert them into output list so that they can be averaged over
	for file in result_files:
		
		file = os.path.join(path, file)
		file = open(file, "rb")
		cur_result = pickle.load(file)
		output.append(cur_result)
		file.close()

	algorithm = output[0].algorithm
	num_clusters = output[0].num_clusters

	# Copy simulation info and algorithm details into average result object
	average_result = Result(algorithm, num_clusters)
	average_result.CopySimulationInfo(output[0])
	average_result.save_path = output[0].save_path
	
	# Add the solution of one of the trials into the average object
	soln = output[0].GetSolution()
	average_result.AddSolution(soln[0], soln[1])
	average_result.dataset_path = os.path.join(dataset_src, dataset, output[0].run_id)

	num_single_axis_plot = len(output[0].single_axis_plot)
	num_secondary_axis_plot = len(output[0].secondary_axis_plot)

	# Initialise data vectors
	costs = []
	single_y_axis = [[] for i in range(num_single_axis_plot)]
	secondary_y1_axis = [[] for i in range(num_secondary_axis_plot)]
	secondary_y2_axis = [[] for i in range(num_secondary_axis_plot)]
	run_time = []

	# Extract the numbers from result instances so that they can be averaged
	for res in output:
		run_time.append(res.getRunningTime())
		costs.append(res.cost[1])
		for i in range(num_single_axis_plot):
			single_y_axis[i].append(res.single_axis_plot[i][1])
		for i in range(num_secondary_axis_plot):
			secondary_y1_axis[i].append(res.secondary_axis_plot[i][1]) 	
			secondary_y2_axis[i].append(res.secondary_axis_plot[i][2])

	# Average the vectors
	run_time = np.mean(run_time)
	costs = np.mean(costs, axis = 0)								# Note that percentages are averaged here but it wil be recomputed later
	if(num_single_axis_plot):
		single_y_axis = np.mean(single_y_axis, axis = 1)
	if(num_secondary_axis_plot):
		secondary_y1_axis = np.mean(secondary_y1_axis, axis = 1)
		secondary_y2_axis = np.mean(secondary_y2_axis, axis = 1)

	average_result.cost = [output[0].cost[0], costs]
	average_result.addRunningTime(run_time)

	# Insert averaged single axis plot results back into the average_result object
	for i in range(num_single_axis_plot):
		average_result.AddSingleAxisPlot([
				output[0].single_axis_plot[i][0],
				single_y_axis[i],
				output[0].single_axis_plot[i][2],
				output[0].single_axis_plot[i][3],
				output[0].single_axis_plot[i][4]
			])

	# Insert averaged secondary axis plot results back into the average_result object
	for i in range(num_secondary_axis_plot):
		average_result.AddSecondaryAxisPlot([
				output[0].secondary_axis_plot[i][0],
				secondary_y1_axis[i],
				secondary_y2_axis[i],
				output[0].secondary_axis_plot[i][3],
				output[0].secondary_axis_plot[i][4],
				output[0].secondary_axis_plot[i][5],
				output[0].secondary_axis_plot[i][6],
			])

	average_result.refresh()

	return average_result


def SKMvsALGIF_Plot(data, dataset, num_clusters, save_path):

	"""
	This module plots the variation of cost and violations as a function of number of clusters for SKM
	and shows where this curve intersects the cost of ALG-IF
	"""
		
	global ALGORITHM, LINE_STYLE, COLOR, MARKER, ALG_DATASET_SIZE

	plt.clf()
	fig, ax = plt.subplots(len(data), 1)

	# Create secondary y axis
	ax2 = []
	for i in range(len(ax)):
		ax2.append(ax[i].twinx())

	fig.set_size_inches(5, 8.5 * len(dataset) / 5)

	# Adjust layout of subplots
	plt.subplots_adjust(bottom = 0.15, top = 0.96, right = 0.82, left = 0.17, hspace = 0.4)

	# Iterate over the datasets and add their corresponding plot to the respective axis
	it = 0

	# fraction of x_axis to be cropped from the right extreme
	if(FAIRNESS_TYPE == 0):
		CROP = 0.93
	else:
		CROP = 0.70

	for dataset_name in dataset:

		skm_result_vector = data[dataset_name]["soft_kmeans_" + str(ALG_DATASET_SIZE)]
		algif_result_vector = data[dataset_name]["individual_fairness_" + str(ALG_DATASET_SIZE)]

		x_axis = []

		# Add values of stiffness parameter
		for i in range(len(skm_result_vector[0].secondary_axis_plot[0][0])):
			x_axis.append(skm_result_vector[0].secondary_axis_plot[0][0][i])

		i = 0
		while(i < len(skm_result_vector) and skm_result_vector[i].num_clusters != num_clusters):
			i += 1
		assert(i != len(skm_result_vector))

		y1_axis = skm_result_vector[i].secondary_axis_plot[0][1]  	
		y1_axis = np.power(y1_axis, 0.5)			# Compute sqrt of LP objective
		y2_axis = skm_result_vector[i].secondary_axis_plot[0][2]
		# Compute alg-if cost / HKM cost 
		IF_axis = [algif_result_vector[i].cost[1][1] / algif_result_vector[i].cost[1][0]] * len(x_axis) 		
		IF_axis = np.power(IF_axis, 0.5) 		# Compute sqrt of LP objective

		x_axis = x_axis[:-int(CROP * len(x_axis))]
		y1_axis = y1_axis[:-int(CROP * len(y1_axis))]
		y2_axis = y2_axis[:-int(CROP * len(y2_axis))]
		IF_axis = IF_axis[:-int(CROP * len(IF_axis))]

		# Search for the first point where cost of SKM is less than IF cost - Point of intersection of SKM cost and ALG-IF cost
		intersection_idx = 0
		while(intersection_idx < len(y1_axis) and y1_axis[intersection_idx] > IF_axis[intersection_idx]): 
			intersection_idx += 1

		# Plot soft kmeans cost and violations vs stiffness parameter
		ax[it].plot(x_axis, y1_axis, label = "SKM Cost", linestyle = LINE_STYLE[1], color = COLOR[2]) 
		ax2[it].plot(x_axis, y2_axis, label = "SKM Violations", linestyle = LINE_STYLE[2], color = COLOR[0]) 
		# ax[it].plot(x_axis, IF_axis, label = "ALG-IF", linestyle = LINE_STYLE[0], color = COLOR[2])
		
		# Plot vertical black line if ALG-IF and SKM intersect
		if(intersection_idx < len(y1_axis)):
			ax[it].axvline(x_axis[intersection_idx], linestyle = LINE_STYLE[0], color = COLOR[3])

			# Mark the point on violations curve where Cost(ALG-IF) = Cost(SKM)
			ax2[it].annotate(r"\boldmath ${perc}\%$".format(perc = str(int(y2_axis[intersection_idx]))), xy = (x_axis[intersection_idx], y2_axis[intersection_idx]), color = "red", 
							xytext = ((2 * x_axis[intersection_idx] / 3 + x_axis[-1] / 3), 2 * y2_axis[intersection_idx] / 3), arrowprops = dict(arrowstyle = '-|>', 
								connectionstyle = 'arc3', facecolor = 'red'), fontsize = 10)

		it += 1

	# Hide axis for inner plots
	for _ax in ax.flat:
		_ax.label_outer()

	# Set subplot titles and axis labels
	for i in range(len(dataset)):
		ax[i].set_title(dataset[i])

	ax[-1].set_xlabel(r"Stiffness paramemter $(\beta)$", fontsize = LABEL_FONT_SIZE)	
	fig.text(0.045, 0.5, r"Relative Clustering Cost (SKM/HKM)", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)
	fig.text(1 - 0.039, 0.5, r"Individual fairness constraint violations (in \%)", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)

	# Set figure title and axis labels
	handles, labels = ax[0].get_legend_handles_labels()
	_handles, _labels = ax2[0].get_legend_handles_labels()
	handles = handles + _handles
	labels = labels + _labels
	fig.legend(handles, labels, loc='lower center', ncol = 3)

	# Save the figure
	fig.savefig(os.path.join(save_path, "skm_vs_algif_" + str(num_clusters) + "_clusters.eps"))

def SoftKMeansUnfairnessPlot(data, dataset, save_path):

	"""
	This module plots the variation of cost and violations as a function of number of clusters for soft kmeans
	"""
		
	global ALGORITHM, LINE_STYLE, COLOR, MARKER, ALG_DATASET_SIZE

	plt.clf()
	fig, ax = plt.subplots(len(data), 1)

	# Create secondary y axis
	ax2 = []
	for i in range(len(ax)):
		ax2.append(ax[i].twinx())

	fig.set_size_inches(5, 8.5 * len(dataset) / 5)

	# Adjust layout of subplots
	plt.subplots_adjust(bottom = 0.14, top = 0.96, right = 0.82, left = 0.17, hspace = 0.4)

	# Iterate over the datasets and add their corresponding plot to the respective axis
	it = 0
	algorithm = "soft_kmeans_" + str(ALG_DATASET_SIZE)
	for dataset_name in dataset:

		result_vector = data[dataset_name][algorithm]
		x_axis = []

		# Add values of stiffness parameter
		for i in range(len(result_vector[0].secondary_axis_plot[0][0])):
			x_axis.append(result_vector[0].secondary_axis_plot[0][0][i])

		# Add the cost vectors for each algorithm for different cluster sizes
		y1_axis = []
		y2_axis = []
		for i in range(len(result_vector)):
			
			y1_axis.append(np.power(result_vector[i].secondary_axis_plot[0][1], 0.5))  	
			y2_axis.append(result_vector[i].secondary_axis_plot[0][2])

		# Plot soft kmeans cost and violations vs stiffness parameter
		c_it = 0		# Style iterator
		for i in range(1, len(result_vector)):
			ax[it].plot(x_axis, y1_axis[i], label = "k = " + str(result_vector[i].num_clusters), linestyle = LINE_STYLE[c_it], color = COLOR[2]) 
			ax2[it].plot(x_axis, y2_axis[i], label = "k = " + str(result_vector[i].num_clusters), linestyle = LINE_STYLE[c_it], color = COLOR[0]) 
			c_it += 1

		it += 1

	# Hide axis for inner plots
	for _ax in ax.flat:
		_ax.label_outer()

	# Set subplot titles and axis labels
	for i in range(len(dataset)):
		ax[i].set_title(dataset[i])

	ax[-1].set_xlabel(r"Stiffness paramemter $(\beta)$", fontsize = LABEL_FONT_SIZE)	
	fig.text(0.045, 0.5, r"Relative Clustering Cost (SKM/HKM)", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)
	fig.text(1 - 0.039, 0.5, r"Individual fairness constraint violations (in \%)", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)

	# Set figure title and axis labels
	handles, labels = ax[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol = 4)

	# Save the figure
	fig.savefig(os.path.join(save_path, "soft_kmeans_unfairness.eps"))

def IFViolationsPlot(data, save_path):

	"""
	This module plots the percentage of IF constraints violated as a function of cluster size
	Input:
		save_path   	- Directory to which result has to be saved
	"""
		
	global SHORT_ALIAS, LINE_STYLE, COLOR, MARKER, DATASET, ALG_DATASET_SIZE

	algorithm = "statistical_fairness_" + str(ALG_DATASET_SIZE)

	plt.clf()
	fig, ax = plt.subplots(1, 1)

	plt.subplots_adjust(bottom = 0.16, top = 0.96, right = 0.96, left = 0.15)

	# Iterate over the datasets and add their corresponding plot to the respective axis
	it = 0
	for dataset_name in DATASET:

		result_vector = data[dataset_name][algorithm]
		x_axis = []
		y_axis = []

		# Populate the values into the corresponding axes
		for i in range(len(result_vector)):
			x_axis.append(result_vector[i].num_clusters)
			y_axis.append(result_vector[i].cost[1][6])

		# Plot hard_cost vs no of clusters
		ax.plot(x_axis, y_axis, label = dataset_name, linestyle = LINE_STYLE[it], marker = MARKER[it], color = COLOR[it])

		it += 1

	ax.set_xlabel(r"Number of clusters $(k)$", fontsize = LABEL_FONT_SIZE)	
	fig.text(0.065, 0.5, r"Individual fairness violations (in \%)", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)

	# Set figure title and axis labels
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, loc='lower right')

	# Save the figure
	fig.savefig(os.path.join(save_path, "sf_unfairness.eps"))

def SFViolationsPlot(data, save_path):

	"""
	This module plots the statistical bias of IF as a function of cluster size
	Input:
		save_path   	- Directory to which result has to be saved
	"""
		
	global SHORT_ALIAS, LINE_STYLE, COLOR, MARKER, DATASET, ALG_DATASET_SIZE

	algorithm = "individual_fairness_" + str(ALG_DATASET_SIZE)

	plt.clf()
	fig, ax = plt.subplots(1, 1)

	# Iterate over the datasets and add their corresponding plot to the respective axis
	it = 0
	for dataset_name in DATASET:

		result_vector = data[dataset_name][algorithm]
		x_axis = []
		bias = []

		# Populate the values into the corresponding axes
		for i in range(len(result_vector)):
			x_axis.append(result_vector[i].num_clusters)
			bias.append(result_vector[i].cost[1][6])

		# Plot hard_cost vs no of clusters
		ax.plot(x_axis, bias, label = dataset_name, linestyle = LINE_STYLE[it], marker = MARKER[it], color = COLOR[it])

		it += 1

	fig.text(0.035, 0.5, "Statistical bias", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)

	# Set figure title and axis labels
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, loc='center left')
	ax.set_xlabel(r"Number of clusters $(k)$", fontsize = LABEL_FONT_SIZE)

	# Save the figure
	fig.savefig(os.path.join(save_path, "if_bias.eps"))

def BiasVSEarthmoverPlot(data, dataset, save_path):

	"""
	This module plots the statistical bias of IF as a function of cluster size
	Input:
	"""
		
	global SHORT_ALIAS, LINE_STYLE, COLOR, MARKER, ALG_DATASET_SIZE

	algorithm = "individual_fairness_" + str(ALG_DATASET_SIZE)

	plt.clf()
	fig, ax = plt.subplots(3, 2, sharex = True)

	fig.set_size_inches(10, 6)
	fig.delaxes(ax[0][1])

	# Adjust layout of subplots
	plt.subplots_adjust(bottom = 0.14, top = 0.93, right = 0.94, left = 0.11, hspace = 0.4)

	# Iterate over the datasets and add their corresponding plot to the respective axis
	it = 0
	for dataset_name in dataset:

		result_vector = data[dataset_name][algorithm]
		x_axis = []
		bias = []
		earthmover = []

		# Populate the values into the corresponding axes
		for i in range(len(result_vector)):
			x_axis.append(result_vector[i].num_clusters)
			bias.append(result_vector[i].cost[1][6])
			earthmover.append(result_vector[i].cost[1][7])

		# Plot hard_cost vs no of clusters
		if(it <= 2):
			x = it
			y = 0
		else:
			x = it - 2
			y = 1
		
		ax[x][y].plot(x_axis, bias, label = "Statistical bias", linestyle = LINE_STYLE[0], marker = MARKER[0], color = COLOR[0])
		ax[x][y].plot(x_axis, earthmover, label = "Earth-mover distance", linestyle = LINE_STYLE[2], marker = MARKER[2], color = COLOR[2])
		ax[x][y].set_title(dataset[it])

		it += 1

	fig.text(0.5, 0.035, r"Number of clusters $(k)$", ha = "center", va = 'center', fontsize = LABEL_FONT_SIZE)
	fig.text(0.035, 0.5, "Statistical bias", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)

	# Set figure title and axis labels
	handles, labels = ax[2][0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.90,0.90))

	# Save the figure
	fig.savefig(os.path.join(save_path, "if_bias_vs_earthmover.eps"))

def SampleAndAssign(assignment, num_trials):

	"""
	Realises an assignment for each point given the distribution over centers. 
	Input:
		assignment: 	2D array denoting the distribution of centres for each point
	Output:
		label 	  :		Class label for each point.
	"""

	num_points = assignment.shape[0]
	num_centers = assignment.shape[1]
	labels = []

	# Generate num_trial samples for each point and choose max frequency center. 
	for i in range(num_points):

		# Add 1e-6 to compensate for precision issues
		assignment[i] += 1e-6
		assignment[i] /= assignment[i].sum()
		out = np.random.choice(num_centers, size = (num_trials), replace = True, p = assignment[i])
		labels.append(np.bincount(out).argmax())

	return np.array(labels)

def pnnDiffClusterPercentage(dataset, labels, num_neighbours):

	"""
	Returns the percentage of pairs within num_neighbours nearest neighbours belonging to different clusters
	Input:
		dataset 		- 2D description of the data
		labels 			- Assignment of points to centers
		num_neighbours 	- Number of nearest neighbours to consider 
	Output:
		percentage 		- Percentage of pairs belonging to different clusters
	"""

	num_samples = dataset.shape[0]
	
	# Build ball_tree for kNN queries
	knn = NearestNeighbors(n_neighbors = num_neighbours + 1, algorithm = "ball_tree").fit(dataset)	# One more than num_neighbours as query point itself is a neighbour which is ignored
	_, neighbours = knn.kneighbors(dataset)
	neighbours = neighbours[:, 1:]			# Remove the first nearest neighbour which is itself

	# Compute knn mask - knn[i][j] = true if order of j wrt i is <= num_neighbours 
	knn = np.zeros([num_samples, num_samples]).astype(bool)
	row_id = np.asarray([i for i in range(num_samples) for j in range(num_neighbours)]).reshape(-1)
	knn[row_id, neighbours.reshape(-1)] = True

	# Compute label similarity matrix
	sim = (labels.reshape(-1, 1) == labels.reshape(1, -1))

	from utilities.distance_metrics import L2Distance
	dist = L2Distance(dataset, True)

	return 100 * (1 - ((knn & sim).sum() / (num_samples * num_neighbours))), knn, dist

def HardAssignmentFairness(data, save_path, algo_subset, num_trials, num_neighbours):

	"""
	Evaluates how fair hard assignment is, to individuals
	Input:
		save_path   	- Directory to which result has to be saved
		algo_subset 	- Algorithms on which the experiment is carried out
		num_trials 		- Number of trials over which max frequency is computed
		num_neighbours 	- Number of nearest neighbours to consider, while evaluating individual fairness violations
	"""

	global DATASET, SHORT_ALIAS, LINE_STYLE, MARKER, COLOR, ALG_DATASET_SIZE

	plt.clf()
	fig, ax = plt.subplots(len(DATASET), 1)
	fig.set_size_inches(5, 8.5 * len(DATASET) / 5)

	# Adjust layout of subplots
	plt.subplots_adjust(bottom = 0.16, top = 0.97, right = 0.89, left = 0.15, hspace = 0.4)

	it = 0

	for dataset_name in DATASET:

		c_it = 0

		for alg_name in algo_subset:

			violation = []
			x_axis = []

			result_vector = data[dataset_name][alg_name]
			for i in range(len(result_vector)):
				
				centres, assignment = result_vector[i].GetSolution()
				dataset_path = result_vector[i].dataset_path
				dataset, groups = LoadPickledData(dataset_path)

				dataset = dataset[: ALG_DATASET_SIZE]
				groups = groups[: ALG_DATASET_SIZE]

				labels = SampleAndAssign(assignment, num_trials)
				perc_violations, knn, dist = pnnDiffClusterPercentage(dataset, labels, num_neighbours)			# Percentage of pairs within p-nn that belong to the same group

				x_axis.append(result_vector[i].num_clusters)
				violation.append(round(perc_violations, 2))
				# print(dataset_name, alg_name, result_vector[i].num_clusters, round(perc_violations, 2), round(result_vector[i].getRunningTime()))

			ax[it].plot(x_axis, violation, label = SHORT_ALIAS[alg_name[:alg_name.rfind("_")]], linestyle = LINE_STYLE[c_it], marker = MARKER[c_it], color = COLOR[c_it])
			c_it += 1

		it += 1

	# Hide axis for inner plots
	for _ax in ax.flat:
		_ax.label_outer()

	# Set subplot titles and axis labels
	for i in range(len(DATASET)):
		ax[i].set_title(DATASET[i])

	ax[-1].set_xlabel(r"Number of clusters $(k)$", fontsize = LABEL_FONT_SIZE)	
	fig.text(0.035, 0.5, r"Percentage Violations", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)

	# Set figure title and axis labels
	handles, labels = ax[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol = 2)

	# Save the figure
	fig.savefig(os.path.join(save_path, "violations_after_sampling.eps"))

def CostVariationPlotSmall(data, save_path, algorithms, datasets, plot_name, title, plotHKM = False, printRatio = False):

	"""
	This module plots the variation of cost as a function of number of clusters for specified algorithms
	This implementation is the same as CostVariationPlot except that the plot presentation attributes are tuned to fit lesser datasets
	Input:
		data 			- Dictionary of result objects
		save_path   	- Directory to which result has to be saved
		datasets 		- <=2 datasets for which sub-plots should appear in the figure
		plot_name		- Name in which the plot is to be saved
		title 			- String denoting plot title
		hard_cost 		- if true, HKM cost is also plotted in the figure
		printRatio 		- Prints the approximation ratio on stdout as well as on the plot (approximation ratio is calculated between the first and last algorithm in the list of algorithms)
	"""
		
	global ALGORITHM, SHORT_ALIAS, LINE_STYLE, COLOR, MARKER

	plt.clf()
	fig, ax = plt.subplots(len(datasets), 1)
	if(len(datasets) == 1):
		ax = [ax]

	if(len(datasets) == 2):
		fig.set_size_inches(5, 3.5)
	elif(len(datasets) == 1):
		fig.set_size_inches(5, 2)
	elif(len(datasets) == 3):
		fig.set_size_inches(5, 4.5)

	if(printRatio):
		print(title)
		print("Cost(ALG) / Cost(OPT), Cost(ALG) / Cost(HKM)")

	# Adjust layout of subplots
	if(len(datasets) == 2):
		plt.subplots_adjust(bottom = 0.34, top = 0.92, right = 0.89, left = 0.15, hspace = 0.4)
	elif(len(datasets) == 1):
		plt.subplots_adjust(bottom = 0.53, top = 0.87, right = 0.89, left = 0.15, hspace = 0.4)
	elif(len(datasets) == 3):
		plt.subplots_adjust(bottom = 0.24, top = 0.92, right = 0.89, left = 0.15, hspace = 0.4)

	# Iterate over the datasets and add their corresponding plot to the respective axis
	it = 0
	for dataset_name in datasets:

		result_vector = data[dataset_name][algorithms[0]]
		x_axis = []
		hard_cost = []

		dataset_size = int(result_vector[0].save_path[result_vector[0].save_path.rfind("_") + 1:])

		# Add hard clustering cost to the corresponding vector
		for i in range(len(result_vector)):
			x_axis.append(result_vector[i].num_clusters)
			hard_cost.append(np.power(result_vector[i].cost[1][0] * dataset_size, 0.5))

		# Add the cost vectors for each algorithm for different cluster sizes
		alg_cost = [[] for i in range(len(algorithms))]
		for i in range(len(algorithms)):
			result_vector = data[dataset_name][algorithms[i]]
			for j in range(len(result_vector)):
				alg_cost[i].append(np.power(result_vector[j].cost[1][1] * dataset_size, 0.5))
			alg_cost[i] = np.asarray(alg_cost[i])

		if(printRatio):
			print("{:40}:".format(dataset_name), end = " ")
			print(round(np.max(alg_cost[0] / alg_cost[-1]), 2), ",", round(np.max(alg_cost[0] / hard_cost), 2))
			# print(np.round(alg_cost[0] / hard_cost, 2))
			# print(np.round(alg_cost[1] / hard_cost, 2))

		if(plotHKM):
			# Plot hard_cost vs no of clusters
			ax[it].plot(x_axis, hard_cost, label = "HKM", linestyle = LINE_STYLE[0], marker = MARKER[0], color = COLOR[0])
			alg_cost.append(hard_cost)

		# Plot alg cost vs no of clusters for every algo except soft k-means
		c_it = 1		# Style iterator
		for i in range(len(algorithms)):
			ax[it].plot(x_axis, alg_cost[i], label = SHORT_ALIAS[algorithms[i][:algorithms[i].rfind("_")]], linestyle = LINE_STYLE[c_it], marker = MARKER[c_it], color = COLOR[c_it]) 
			c_it += 1

		if(printRatio):
			width = max(np.max(alg_cost[0]), np.max(alg_cost[-1])) - min(np.min(alg_cost[0]), np.min(alg_cost[-1]))
			y_shift = width * 0.2
			for i in range(len(x_axis)):
				ax[it].annotate(str(round(alg_cost[0][i] / alg_cost[-1][i], 2)), xy = (x_axis[i] - 0.2, alg_cost[0][i] - y_shift), 
							color = "black", fontsize = 10)

		it += 1

	if(printRatio):
		print()

	if(len(datasets) > 1):
		# Hide axis for inner plots
		for _ax in ax.flat:
			_ax.label_outer()

	# Set subplot titles and axis labels
	for i in range(len(datasets)):
		ax[i].set_title(datasets[i])

	ax[-1].set_xlabel(r"Number of clusters $(k)$", fontsize = LABEL_FONT_SIZE - 2)	
	fig.text(0.037, 0.6, r"Clustering Cost", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE - 2)

	# Set figure title and axis labels
	handles, labels = ax[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol = 3, fontsize = AXIS_FONT_SIZE)

	# Save the figure
	fig.savefig(os.path.join(save_path, "elbow_curve_" + plot_name + "small.eps"))

def CostVariationPlot(data, save_path, algorithms, plot_name, title, plotHKM = False, printRatio = False):

	"""
	This module plots the variation of cost as a function of number of clusters for specified algorithms
	Input:
		data 			- Dictionary of result objects
		save_path   	- Directory to which result has to be saved
		plot_name		- Name in which the plot is to be saved
		title 			- String denoting plot title
		hard_cost 		- if true, HKM cost is also plotted in the figure
		printRatio 		- Prints the approximation ratio on stdout as well as on the plot (approximation ratio is calculated between the first and last algorithm in the list of algorithms)
	"""
		
	global ALGORITHM, SHORT_ALIAS, LINE_STYLE, COLOR, MARKER, DATASET

	plt.clf()
	fig, ax = plt.subplots(len(DATASET), 1)
	fig.set_size_inches(5, 8.5 * len(DATASET) / 5)

	if(printRatio):
		print(title)
		print("Cost(ALG) / Cost(OPT), Cost(ALG) / Cost(HKM), Cost(OPT) / Cost(HKM)")

	# Adjust layout of subplots
	plt.subplots_adjust(bottom = 0.14, top = 0.96, right = 0.89, left = 0.17, hspace = 0.4)

	# Iterate over the datasets and add their corresponding plot to the respective axis
	it = 0
	for dataset_name in DATASET:

		result_vector = data[dataset_name][algorithms[0]]
		x_axis = []
		hard_cost = []

		dataset_size = int(result_vector[0].save_path[result_vector[0].save_path.rfind("_") + 1:])

		# Add hard clustering cost to the corresponding vector
		for i in range(len(result_vector)):
			x_axis.append(result_vector[i].num_clusters)
			hard_cost.append(np.power(result_vector[i].cost[1][0] * dataset_size, 0.5))

		# Add the cost vectors for each algorithm for different cluster sizes
		alg_cost = [[] for i in range(len(algorithms))]
		for i in range(len(algorithms)):
			result_vector = data[dataset_name][algorithms[i]]
			for j in range(len(result_vector)):
				alg_cost[i].append(np.power(result_vector[j].cost[1][1] * dataset_size, 0.5))
			alg_cost[i] = np.asarray(alg_cost[i])

		if(printRatio):
			print("{:40}:".format(dataset_name), end = " ")
			print(round(np.max(alg_cost[0] / alg_cost[-1]), 2), ",", round(np.max(alg_cost[0] / hard_cost), 2), ",", round(np.max(alg_cost[-1] / hard_cost), 2))
			# print(np.round(alg_cost[0] / hard_cost, 2))
			# print(np.round(alg_cost[1] / hard_cost, 2))

		if(plotHKM):
			# Plot hard_cost vs no of clusters
			ax[it].plot(x_axis, hard_cost, label = "HKM", linestyle = LINE_STYLE[0], marker = MARKER[0], color = COLOR[0])
			alg_cost.append(hard_cost)

		# Plot alg cost vs no of clusters for every algo except soft k-means
		c_it = 1		# Style iterator
		for i in range(len(algorithms)):
			ax[it].plot(x_axis, alg_cost[i], label = SHORT_ALIAS[algorithms[i][:algorithms[i].rfind("_")]], linestyle = LINE_STYLE[c_it], marker = MARKER[c_it], color = COLOR[c_it]) 
			c_it += 1

		if(printRatio):
			width = max(np.max(alg_cost[0]), np.max(alg_cost[-1])) - min(np.min(alg_cost[0]), np.min(alg_cost[-1]))
			y_shift = width * 0.2
			for i in range(len(x_axis)):
				ax[it].annotate(str(round(alg_cost[0][i] / alg_cost[-1][i], 2)), xy = (x_axis[i] - 0.2, alg_cost[0][i] - y_shift), 
							color = "black", fontsize = 10)

		it += 1

	if(printRatio):
		print()

	# Hide axis for inner plots
	for _ax in ax.flat:
		_ax.label_outer()

	# Set subplot titles and axis labels
	for i in range(len(DATASET)):
		ax[i].set_title(DATASET[i])

	ax[-1].set_xlabel(r"Number of clusters $(k)$", fontsize = LABEL_FONT_SIZE)	
	fig.text(0.042, 0.5, r"Clustering Cost", ha = "center", va = 'center', rotation = 'vertical', fontsize = LABEL_FONT_SIZE)

	# Set figure title and axis labels
	handles, labels = ax[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol = 3, fontsize = AXIS_FONT_SIZE)

	# Save the figure
	fig.savefig(os.path.join(save_path, "elbow_curve_" + plot_name + ".eps"))

def PrintALGIFTime(path):

	"""
	Prints the time taken, as specified in result files in `path` folder
	"""

	files = os.listdir(path)

	for file in files:

		f = open(os.path.join(path, file), "rb")
		result = pickle.load(f)

		print(file, round(result.getRunningTime()))

if(__name__ == "__main__"):

	utils.ResetWorkspace()

	data = {}
	os.system("rm -rf " + INFERENCE_SAVE_PATH)
	matplotlib.rcParams['text.usetex'] = True
	font = {
		'family' : 'serif',
		'size'   : 13
	}
	matplotlib.rc('font', **font)

	ALGORITHM = [
		"soft_kmeans_" + str(ALG_DATASET_SIZE), 
		"individual_fairness_" + str(ALG_DATASET_SIZE), 
		"statistical_fairness_" + str(ALG_DATASET_SIZE), 
		"statistical_individual_fairness_" + str(ALG_DATASET_SIZE), 
		"individual_fairness_" + str(OPT_DATASET_SIZE),
		"statistical_individual_fairness_" + str(OPT_DATASET_SIZE), 
		"opt_individual_fairness_" + str(OPT_DATASET_SIZE), 
		"opt_statistical_individual_fairness_" + str(OPT_DATASET_SIZE)
	] 	# <soft_kmeans/individual_fairness/statistical_fairness/statistical_individual_fairness/opt_individual_fairness/opt_statistical_individual_fairness>


	for dataset in DATASET:
		data[dataset] = {}
		
		for algorithm in ALGORITHM:

			data[dataset][algorithm] = []

			path = os.path.join(DUMP_PATH, dataset)
			path = os.path.join(path, algorithm)

			# Extract the result files for every trial	
			cluster_files = os.listdir(path)
			cluster_files.sort(key=lambda item: (len(item), item))

			for i in range(len(cluster_files)):

				target = os.path.join(path, cluster_files[i])
				# Average the result objects over multiple trials
				result = computeAverage(target, dataset, PICKLED_DATASET)

				# Save the costs into a text file in ./Inferences/<dataset>/<algorithm>/<cluster_size>/cost.txt
				save_path = os.path.join(INFERENCE_SAVE_PATH, dataset)
				save_path = os.path.join(save_path, algorithm)
				save_path = os.path.join(save_path, cluster_files[i])
				result.save(dataset, save_path)

				data[dataset][algorithm].append(result)

	# Clean existing plots and generate fresh plots
	os.system("rm -rf " + INFERENCE_SAVE_PATH + "/*.*")

	# Datasets on which cost analysis should be done. 
	dataset_subset = ["adult", "creditcard"]
	algo_subset = [
		"soft_kmeans_" + str(ALG_DATASET_SIZE),
		"individual_fairness_" + str(ALG_DATASET_SIZE),
		"statistical_fairness_" + str(ALG_DATASET_SIZE),
		"statistical_individual_fairness_" + str(ALG_DATASET_SIZE)
	]

	font = {
		'size'  : AXIS_FONT_SIZE
	}
	matplotlib.rc('font', **font)

	# Clustering Cost vs Num Clusters | ALG-IF and OPT-IF | plot - 1
	algos_of_interest = ["individual_fairness_" + str(OPT_DATASET_SIZE), "opt_individual_fairness_" + str(OPT_DATASET_SIZE)]
	CostVariationPlot(data, INFERENCE_SAVE_PATH, algos_of_interest, "ALG-IF-OPT-IF", "Performance of ALG-IF", plotHKM = True, printRatio = False)
	# CostVariationPlotSmall(data, INFERENCE_SAVE_PATH, algos_of_interest, dataset_subset, "ALG-IF-OPT-IF", "Performance of ALG-IF", plotHKM = True, printRatio = False)

	# Clustering Cost vs Num Clusters | ALG-CF and OPT-CF | plot - 2 
	algos_of_interest = ["statistical_individual_fairness_" + str(OPT_DATASET_SIZE), "opt_statistical_individual_fairness_" + str(OPT_DATASET_SIZE)]
	CostVariationPlot(data, INFERENCE_SAVE_PATH, algos_of_interest, "ALG-CF-OPT-CF", "Performance of ALG-CF", plotHKM = True, printRatio = False)
	# CostVariationPlotSmall(data, INFERENCE_SAVE_PATH, algos_of_interest, dataset_subset, "ALG-CF-OPT-CF", "Performance of ALG-CF", plotHKM = True, printRatio = False)

	# Clustering Cost and Violations vs Stiffness Parameter for k-means | plot 3
	SoftKMeansUnfairnessPlot(data, DATASET, INFERENCE_SAVE_PATH)

	# SKM vs ALG-IF | plot 4
	for k in range(2, 11, 2):
		SKMvsALGIF_Plot(data, DATASET, k, INFERENCE_SAVE_PATH)

	# IF violations of GF algorithm | plot 5
	IFViolationsPlot(data, INFERENCE_SAVE_PATH)

	# SF violations of ALG-IF algorithm | plot 6
	SFViolationsPlot(data, INFERENCE_SAVE_PATH)

	# Bias vs Earthmover - ALG-IF algorithm | plot 7
	BiasVSEarthmoverPlot(data, DATASET, INFERENCE_SAVE_PATH)

	dataset_subset = ["creditcard"]
	# Clustering Cost vs Num Clusters | HKM, OPT-IF, OPT-CF | plot - 8
	algos_of_interest = ["opt_individual_fairness_" + str(OPT_DATASET_SIZE), "opt_statistical_individual_fairness_" + str(OPT_DATASET_SIZE)]
	CostVariationPlot(data, INFERENCE_SAVE_PATH, algos_of_interest, "OPT-IF-CF-HKM", "Performance of OPT-IF and OPT-CF", plotHKM = True, printRatio = False)
	# CostVariationPlotSmall(data, INFERENCE_SAVE_PATH, algos_of_interest, dataset_subset, "OPT-IF-CF-HKM", "Performance of OPT-IF and OPT-CF", plotHKM = True, printRatio = False)

	# Measuring individual fairness after sampling from the produced distribution
	HardAssignmentFairness(data, INFERENCE_SAVE_PATH, algo_subset, 100, NUM_NEIGHBOURS)
	
	# Prints the time taken for different datasets
	PrintALGIFTime(TIMING_RESULT_PATH)
