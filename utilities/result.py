# File: result.py

import os
import pickle
import numpy as np
from copy import deepcopy
from utilities.utils import SingleAxisPlot, SecondaryAxisPlot

class Result():

	"""
	A class to store the results of a single run of a simulation 
	"""

	def __init__(self, algorithm, num_clusters, run_id = 0, save_path = None):

		self.algorithm = algorithm				# Name of the clustering algorithm
		self.num_clusters = num_clusters		# Number of clusters
		self.run_id = run_id					# The algorithm is run NUM_SETS times, each set containing 1000 uniform random samples
		self.save_path = save_path
		self.running_time = 0

		self.sim_info = []						# every tuple in the list is printed on a new line and every element of the tuple is printed with spaces 
		self.sim_info.append([self.algorithm + " Simulation Parameters"])

		self.cost = [[], []]							# List of two tuples, the first one containing cost description and the second one contains numerical cost value
		self.single_axis_plot = []
		self.secondary_axis_plot = []
		self.solution = None
		self.allowRefresh = True

	def __str__(self):

		"""
		Printing the result object ends up printing the simulation parameters associated with the run
		"""

		# Format the headings properly
		for i in range(1, len(self.sim_info)):
			self.sim_info[i][0] = self.sim_info[i][0].replace("\t", "")
			self.sim_info[i][0] = self.sim_info[i][0].replace("  ", " ")
			self.sim_info[i][0] = self.sim_info[i][0].replace(":", "")
			self.sim_info[i][0] = "{:40} :".format(self.sim_info[i][0])

		res = ""
		for line in self.sim_info:
			for item in line:
				res += item + " "
			res += "\n"

		return res

	def getRunningTime(self):

		"""
		Returns the time taken by the algorithm
		"""

		return self.running_time

	def addRunningTime(self, t):

		"""
		Time taken by the clustering algorithm
		"""

		self.running_time = t

	def AddSimulationInfo(self, heading, value):

		"""
		Adds a new line with format - heading <space> value - to the list of simulation parameters
		"""

		self.sim_info.append([heading, str(value)])

	def CopySimulationInfo(self, _result):

		"""
		Copy constructor that copies simulation info from _result to self
		"""

		self.sim_info = deepcopy(_result.sim_info)

	def AddCost(self, heading, value):

		"""
		Adds a new simulation cost entry - This is averaged over various result instances with same simulation parameters
		The first 5 values in this vector corresponds to the following:
			- Hard cost
			- Alg cost
			- Uniform cost
			- Percentage loss of hard cost wrt Alg cost
			- Percentage gain of uniform cost wrt Alg cost
		There are more values depending on the algorithn used
		"""

		self.cost[0].append(heading)
		self.cost[1].append(value)

	def refresh(self):

		"""
		This function is invoked after averaging the results over multiple trials (in inference.py only). It is responsible for computing the p^th 
		root of the cost function (The optimisation function itself did not contain the p^th root because of linearity requirrement) 
		and recalculating percentages.
		"""

		# Refreshing is allowed only once because it computes the p^th root of the cost vectors. (p = 2 in our case) 
		if(not self.allowRefresh):
			return

		self.allowRefresh = False

		# Compute the sqrt (p^th root for p = 2) of HKM, ALG and Uniform cost
		self.cost[1][:2] = np.power(self.cost[1][:2], 0.5)

		# Converts perc decrease metric stored in the y1 axis of secondary axis plot to Cost(SKM) / Cost(HKM) - Applicable only for kmeans
		if(len(self.secondary_axis_plot)): 		# True only for kmeans result object
			
			perc_decrease = self.secondary_axis_plot[0][1]

			# Convert perc_decrease to SKM / HKM. perc_decrease = 100 * (SKM - HKM) / SKM
			inv_ratio = 1 - (perc_decrease / 100)
			ratio = 1 / inv_ratio

			# Compute the sqrt of ratio as costs are being raised to p^th root (p = 2)
			self.secondary_axis_plot[0][1] = np.power(ratio, 0.5)

		# Recompute cost loss of HKM and cost gain of Uniform w.r.t. alg cost
		self.cost[1][3] = round(((self.cost[1][1] - self.cost[1][0]) / self.cost[1][1]) * 100, 2)
		self.cost[1][4] = round(((self.cost[1][2] - self.cost[1][1]) / self.cost[1][1]) * 100, 2)

	def GetCost(self):

		"""
		Pretty printing tool
		"""

		# Format headings properly
		for i in range(len(self.cost[0])):
			self.cost[0][i] = self.cost[0][i].replace("\t", "")
			self.cost[0][i] = self.cost[0][i].replace("  ", " ")
			self.cost[0][i] = self.cost[0][i].replace(":", "")
			self.cost[0][i] = "{:40} :".format(self.cost[0][i])

		cost = "Algorithm Cost\n"
		for i in range(len(self.cost[0])):
			cost += self.cost[0][i] + " " + str(round(self.cost[1][i], 2)) + "\n"

		return cost

	def AddSecondaryAxisPlot(self, data):

		"""
		Adds values corresponding to secondary axis plot - This will be averaged over various result instances
		Format:
			x_axis data
			y1_axis data
			y2_axis data
			x_label
			y1_label
			y2_label
			title
		"""

		self.secondary_axis_plot.append(data)

	def AddSingleAxisPlot(self, data):

		"""
		Adds values corresponding to primary axis plot - This will be averaged over various result instances
		Format:
			x_axis data
			y_axis data
			x_label
			y_label
			title
		"""

		self.single_axis_plot.append(data)

	def GetSolution(self):

		"""
		Returns the proposed cluster centres along with the obtained probability distribution for each point
		"""

		return self.solution

	def AddSolution(self, centres, distribution):

		"""
		Adds the proposed cluster centres along with the obtained probability distribution for every point
		"""

		self.solution = (centres, distribution)

	def dump(self):

		"""
		Dumps the entire result instance into the disk
		"""

		if(self.save_path is None):
			print("Save path is not set... returning")
			return

		target = os.path.join(self.save_path, str(self.num_clusters) + "_clusters")
		try:
			os.makedirs(target)
		except:
			print("[ " + target + " directory already exists - Ignoring exception")
			pass

		# Converting float lists to numpy arrays
		self.cost[1] = np.asarray(self.cost[1])

		# Dump the entire instance of Result object 
		print("Dumping results of " + self.algorithm + " dataset " + self.run_id)
		target = os.path.join(target, self.algorithm + "_" + self.run_id + ".result")
		file = open(target, "wb")
		pickle.dump(self, file)
		file.close()
		print("Dump complete!")

	def save(self, dataset, path):

		"""
		Saves the plots and generated cost vectors into human readable format - .jpg/.txt
		"""

		# Create target folder if it doesn't exist. Clear contents if it does
		try:
			os.makedirs(path)
		except:
			os.system("rm -rf " + path + "/*")

		# Write the average cost to cost.txt
		target = os.path.join(path, "cost.txt")
		file = open(target, "w")
		file.write(self.__str__() + "\n")
		file.write(self.GetCost())
		file.close()

		# Generate Single Axis Plots 
		for i in range(len(self.single_axis_plot)):
			target = os.path.join(path, self.single_axis_plot[i][4] + ".jpg")
			SingleAxisPlot(
					self.single_axis_plot[i][0],
					self.single_axis_plot[i][1],
					target,
					self.single_axis_plot[i][2],
					self.single_axis_plot[i][3],
					self.single_axis_plot[i][4] + " - " + dataset + " dataset, " + str(self.num_clusters) + " clusters"
				)

		# Generate Secondary Axis Plots
		for i in range(len(self.secondary_axis_plot)):
			target = os.path.join(path, self.secondary_axis_plot[i][6] + ".jpg")
			SecondaryAxisPlot(
					self.secondary_axis_plot[i][0],
					self.secondary_axis_plot[i][1],
					self.secondary_axis_plot[i][2],
					target,
					self.secondary_axis_plot[i][3],
					self.secondary_axis_plot[i][4],
					self.secondary_axis_plot[i][5],
					self.secondary_axis_plot[i][6] + " - " + dataset + " dataset, " + str(self.num_clusters) + " clusters"
				) 

if(__name__ == "__main__"):

	pass
