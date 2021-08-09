# simulated annealing a n-dimensional objective function
from numpy import mean
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy import asarray
from numpy.random import seed
from matplotlib import pyplot
import pandas as pd
import Fitting as Fit
import Gillespie as Glp
import os
import pickle

class Simulated_Annealing():

	def __init__(self):
		# initialise constants
		self.species_lenser = ["x","y","z"]  # z inner und x free, y loosely bound
		# loss side
		self.L_lenser = pd.DataFrame({"reaction01":[0, 0, 0],  # außen - x
					"reaction02":[1, 0, 0],  # x - y
					"reaction03":[0, 1, 0],  # y - z
					"reaction11":[1, 0, 0],  # x - außen
					"reaction12":[0, 1, 0],  # y - x
					"reaction13":[0, 0, 1]},  # z - y
					index = self.species_lenser)
		# gain side
		self.R_lenser = pd.DataFrame({"reaction01":[1, 0, 0],  # außen - x
					"reaction02":[0, 1, 0],  # x - y
					"reaction03":[0, 0, 1],  # y - z
					"reaction11":[0, 0, 0],  # x - außen
					"reaction12":[1, 0, 0],  # y - x
					"reaction13":[0, 1, 0]},  # z - y
					index = self.species_lenser)
		self.N_lenser = self.R_lenser - self.L_lenser

		# number of fluorescent molecules after bleach
		self.startQuantities_lenser = {"x":[0],"y":[0],"z":[0]}

		self.time_limit_lenser = 1160.0  # see time axis of measured data
		self.runs_lenser = 40
		# colours for plotting
		self.colours_lenser = {"x":"red", "y":"forestgreen", "z":"lightblue"}
		# component of the measured FRAP data to be fittet to
		self.pml_component = "PML I WT"

		self.measurement = pd.read_csv("./../Daten/FRAP_comparison_all_new_AssemblyDynamicsOfPMLNuclearBodiesInLivingCells_cleaned.csv", index_col="time[s]")
		self.measured_times = self.measurement.index.tolist()
		self.time_intervals = Fit.divide_time_axis_equidistantly(self.measured_times)

		self.storage_path = './../Daten/SimulatedAnnealing/WeidtkampPeters_Lenser/'
		self.track_data =[]

	# objective function
	def objective_function(self, parameters):
		if len(parameters) != len(self.L_lenser.columns):
			raise IndexError('The number of parameters has to equal the number of reactions in the given reaction network.')
		constants = {reaction:parameters[i] for i, reaction in enumerate(self.L_lenser.columns)}
		# TODO: delete print
		# print(constants)
		# simulate one or more Gillespie runs
		trajectories = Glp.monte_carlo_gillespie(constants, self.L_lenser, self.N_lenser, self.startQuantities_lenser, runs=self.runs_lenser, time_max=self.time_limit_lenser)
		# list of tuples of gillespie times_list and added_quantities_list for output signal
		list_of_output_data = Glp.make_output_signal(trajectories, self.species_lenser)
		# DataFrame of averaged gillespie data per time interval, time intervals as index
		assigned_simulation_data = Fit.assign_simulation_times_to_time_ranges_average(self.time_intervals,self.measured_times,list_of_output_data)
		# normalize the simulation data
		norm_value = 273  # max 200 molecules in a ROI
		normalized_assigned_simulation_data = assigned_simulation_data/norm_value
		# calculate the absolute differences between the simulated and measured data
		differences = Fit.calculate_differences(self.measurement[self.pml_component], normalized_assigned_simulation_data)
		# create a quality score by adding up the differences per individual run and averaging the sums
		quality_of_fitness = mean(differences.sum())
		self.track_data = [trajectories, normalized_assigned_simulation_data, differences, quality_of_fitness]
		
		return quality_of_fitness

	# simulated annealing algorithm
	def simulated_annealing(self, objective, bounds, n_iterations, step_size, temp):
		# create a path to save all files, if needed
		if not os.path.exists(self.storage_path):
			os.makedirs(self.storage_path)
		# generate an initial solution
		best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		# evaluate the initial solution
		best_eval = objective(best)
		# store simulation data
		storage_file = self.storage_path + 'simulation-1_data_pickle_binary'
		with open(storage_file, 'bw') as dumpfile:
			pickle.dump(self.track_data, dumpfile)
		# current working solution
		working, working_eval = best, best_eval
		best_scores = []
		# run simulated annealing
		for i in range(n_iterations):
			# get another solution
			candidate = working + randn(len(bounds)) * step_size
			# the rate constant values of the candidate shall be between 0 (do not take place) and 1 (fully take place)
			for candi in range(len(candidate)):
				if candidate[candi] < 0:
					candidate[candi] = 0.0
				if candidate[candi] > 1:
					candidate[candi] = 1.0
			if sum(candidate) == 0:
				print("The combination of the input parameters has resulted in the overall reactivity of the system a0 being 0. The simulation run %d was therefore skipped." % i)
				continue
			# evaluate candidate solution
			candidate_eval = objective(candidate)
			# store simualtion data
			storage_file = self.storage_path + 'simulation+' + str(i) + '_data_pickle_binary'
			with open(storage_file, 'bw') as dumpfile:
				pickle.dump(self.track_data, dumpfile)
			# check, if the candidate is better than the current best
			if candidate_eval < best_eval:
				# store new best solution
				best, best_eval = candidate, candidate_eval
				# keep track of scores
				best_scores.append(best_eval)
				# report progress
				print('>%d eval(%s) = %.5f' % (i, best, best_eval))
			else:
				print('>%d' % i)
			# difference between candidate and working solution evaluation
			diff = candidate_eval - working_eval
			# calculate temperature for current epoch
			t = temp / float(i + 1)
			# calculate metropolis acceptance criterion
			metropolis = exp(-diff / t)
			# check if we should keep the new solution
			if diff < 0 or rand() < metropolis:
				# store the new working solution
				working, working_eval = candidate, candidate_eval
		return [best, best_eval, best_scores]

def run_simulated_annealing():
	# seed the pseudorandom number generator
	# seed(1)
	# object
	obj = Simulated_Annealing()
	# create ranges for the number of parameters to be optimised
	bounds = asarray([[0.0, 1.0]]*len(obj.L_lenser.columns))
	# number of annealing iterations
	n_iterations = 100
	# maximum step size
	step_size = 0.1
	# initial temperature
	temperature = 50
	# perform the simulated annealing search
	best, score, scores = obj.simulated_annealing(obj.objective_function, bounds, n_iterations, step_size, temperature)
	print('Done!')
	print('f(%s) = %f' % (best, score))
	# line plot of best scores
	pyplot.plot(scores, '.-')
	pyplot.xlabel('Improvement Number')
	pyplot.ylabel('Evaluation objective(c)')
	pyplot.show()

	return obj
