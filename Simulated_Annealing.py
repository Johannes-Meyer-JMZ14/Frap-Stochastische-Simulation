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
		self.runs_lenser = 2
		# colours for plotting
		self.colours_lenser = {"x":"red", "y":"forestgreen", "z":"lightblue"}
		# component of the measured FRAP data for fitting
		self.pml_component = "PML I WT"

		self.measurement = pd.read_csv("./../Daten/FRAP_comparison_all_new_AssemblyDynamicsOfPMLNuclearBodiesInLivingCells_cleaned.csv", index_col="time[s]")
		self.measured_times = self.measurement.index.tolist()
		self.time_intervals = Fit.divide_time_axis_equidistantly(self.measured_times)

		self.track_data =[]

	# objective function
	def objective_function(self, parameters):
		constants = {reaction:parameters[i] for i, reaction in enumerate(self.L_lenser.columns)}
		# simulate one or more Gillespie runs
		trajectory = Glp.monte_carlo_gillespie(constants, self.L_lenser, self.N_lenser, self.startQuantities_lenser, runs=self.runs_lenser, time_max=self.time_limit_lenser)
		# list of tuples of gillespie times_list and added_quantities_list for output signal
		list_of_output_data = Glp.make_output_signal(trajectory, self.species_lenser)
		# DataFrame of averaged gillespie data per time interval, time intervals as index
		assigned_simulation_data = Fit.assign_simulation_times_to_time_ranges_average(self.time_intervals,self.measured_times,list_of_output_data)
		# normalize the simulation data
		norm_value = 200  # max 200 molecules in a ROI
		normalized_assigned_simulation_data = assigned_simulation_data/norm_value
		# calculate the absolute differences between the simulated and measured data
		differences = Fit.calculate_differences(self.measurement[self.pml_component], normalized_assigned_simulation_data)
		# create a quality score by adding up the differences per individual run and averaging the sums
		quality_of_fitness = mean(differences.sum())
		self.track_data.append([trajectory, list_of_output_data, normalized_assigned_simulation_data, differences, quality_of_fitness])
		
		return quality_of_fitness

	# simulated annealing algorithm
	def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
		# generate an initial solution
		best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
		# evaluate the initial solution
		best_eval = objective(best)
		# current working solution
		current, current_eval = best, best_eval
		scores = []
		# run the algorithm
		for i in range(n_iterations):
			# get another solution
			candidate = current + randn(len(bounds)) * step_size
			# the rate constant values of the candidate must not be negative
			for c in candidate:
				if c < 0:
					c = 0
			# evaluate candidate solution
			candidate_eval = objective(candidate)
			# check, if the candidate is better than the current best
			if candidate_eval < best_eval:
				# store new best point
				best, best_eval = candidate, candidate_eval
				# keep track of scores
				scores.append(best_eval)
				# report progress
				print('>%d f(%s) = %.5f' % (i, best, best_eval))
			# difference between candidate and current point evaluation
			diff = candidate_eval - current_eval
			# calculate temperature for current epoch
			t = temp / float(i + 1)
			# calculate metropolis acceptance criterion
			metropolis = exp(-diff / t)
			# check if we should keep the new point
			if diff < 0 or rand() < metropolis:
				# store the new current point
				current, current_eval = candidate, candidate_eval
		return [best, best_eval, scores]

	def run_simulated_annealing(self):
		# seed the pseudorandom number generator
		# seed(1)
		# define range for input
		bounds = asarray([[0.0, 1.0]]*len(self.L_lenser.columns))
		# number of annealing iterations
		n_iterations = 1000
		# maximum step size
		step_size = 0.5
		# initial temperature
		temperature = 50
		# perform the simulated annealing search
		best, score, scores = self.simulated_annealing(self.objective_function, bounds, n_iterations, step_size, temperature)
		print('Done!')
		print('f(%s) = %f' % (best, score))
		# line plot of best scores
		pyplot.plot(scores, '.-')
		pyplot.xlabel('Improvement Number')
		pyplot.ylabel('Evaluation f(x)')
		pyplot.show()