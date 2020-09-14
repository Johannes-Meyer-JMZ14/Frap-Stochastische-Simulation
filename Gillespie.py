import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from copy import deepcopy

# Comments partially cited from "Exact Stochastic Simulation of Coupled Chemical Reactions" by Daniel T. Gillespie in 1977

class Gillespie():

    def __init__(self, L, N, rConstants, quantities):
        # Gillespie Step 0
        # Input the desired values for the M reaction constants c1,...,cM and the N initial molecular population numbers X1,...,XN.
        # stoichiometric matrices as pandas.DataFrame
        self.L = L
        self.N = N
        # stochastic rate constants
        self.rConstants = rConstants
        # initial amounts of molekules in the system
        self.quantities = deepcopy(quantities)
        # time variables for plotting and terminating
        self.times = [0.0]
        self.tmax = 0.0
        # internal variables for the gillespie algorithm
        self.h = dict()
        self.a = dict()
        self.a0 = 0.0

    def run_time_sec(self, tmax):
        """Run single Gillespie steps until tmax is reached."""

        if tmax is None:
            raise ValueError("The time limit was Null.")
        elif not isinstance(tmax, float):
            raise ValueError("The reaction time limit has to be a float.")
        elif tmax < 0:
            raise ValueError("The time limit could not be negative.")

        self.tmax = tmax

        while self.times[-1] < tmax:
            self.run_single_step()

    def run_n_reactions(self, nreactions):
        """Run single Gillespie steps until n reactions are computed."""

        if nreactions is None:
            raise ValueError("The reaction limit was Null.")
        elif not isinstance(nreactions, int):
            raise ValueError("The reaction limit has to be an integer.")
        elif nreactions < 0:
            raise ValueError("The reaction limit could not be negative!")

        for x in range(nreactions):
            self.run_single_step()
    
    def run_single_step(self):
        """Execute the Gillespie algorithm as described in "Exact Stochastic Simulation of Coupled Chemical Reactions" by Daniel T. Gillespie in 1977."""

        # Gillespie Step 1
        # Calculate and store the M quantities a1 = h1c1,..., aM = hMcM for the currnt molecular population numbers, where h_nu is that function of X1,...,XN defined in (15).
        for reaction in self.L.keys():
            self.h[reaction] = self.calc_h(reaction)
            self.a[reaction] = self.calc_a(reaction)
        self.a0 = self.calc_a0()

        # Gillespie Step 2
        # Generate two random numbers r1 and r2 using the unit-interval uniform random number generator, and calculate tau and mu according to (21a) and (21b).
        # Gillespie Step 3
        # Using the tau and mu values obtained in step 2, increase t by tau and adjust the molecular population levels to reflect the occurrence of one R_mu reaction.
        # pick a reaction random
        next_reaction = self.choose_next_reaction()
        # advance time random
        self.advance_time()
        # alter molecule amounts
        self.trace_molecule_changes(next_reaction)

    def binomial_coefficient(self, n, k):

        if 2*k > n:
            k = n-k
        res = 1
        for i in range(1, k+1):
            res = res * (n - k + i) / i
        return res

    def get_reactants(self, reaction):
        """Returns dict = {reactant: quantity}. The reactants (and their quantity) taking part in the specified reaction."""

        reactants = dict()
        # for any analyte in reaction
        for row_key in self.L[reaction].keys():
            # get number and type of analytes used in the reaction
            entry = self.L[reaction][row_key]
            if not isinstance(entry, np.int64):
                raise ValueError("Values of stoichiometric matrices have to be int64.")
            if entry < 0:
                raise ValueError("Values of stoichiometric matrix L must not be negative.")
            # if some analyte will be used
            if entry > 0:
                reactants[row_key] = entry
        return reactants

    def calc_h(self, reaction):
        """Calculates the number of possible combinations of molecules concerning the given reaction."""

        if not isinstance(reaction, str):
            raise ValueError("reaction parameter for calc_h() has to be a string.")
        if reaction not in self.L.keys():
            if reaction == "":
                raise KeyError("Reaction not found, empty key.")
            elif reaction == None:
                raise KeyError("Reaction not found, key is of type " + str(type(reaction)) + ".")
            raise KeyError("Reaction not found.")
        
        # get all reactants taking part in reaction "reaction"
        reactants = self.get_reactants(reaction)  # type is dictionary
        
        ret = 1  # number of possible molecule combinations leading to a reaction
        for species in reactants:
            n = self.quantities[species][-1]  # current amount of molecules of this species, if zero "ret" will be zero
            k = reactants[species]  # number of molecules needed to do the reaction
            # binomial coefficient, number of possibilities to draw k molecules out of n (from one species)
            if k == 1:
                ret = ret * n
            elif k == 2:
                ret = ret * 0.5*n*(n-1)
            elif k == 3:
                ret = ret * (n*(n-1)*(n-2))/6
            else:
                ret = ret * self.binomial_coefficient(n, k)

        return ret

    def calc_a(self, reaction):
        """Multiplies the possible combinations and the stochastical rate constant for all reactions."""

        # reaction must be a string, key of dict and must not be referencing to a negative value
        if not isinstance(reaction, str):
            raise ValueError("reaction parameter for calc_a() has to be a string.")
        if reaction not in self.rConstants.keys():
            raise KeyError("reaction parameter not in rConstants.")
        if reaction not in self.h.keys():
            raise KeyError("reaction parameter not in dictionary h.")
        if self.rConstants[reaction] < 0:
            raise ValueError("Reaction constant for" + reaction + "must not be negative.")
        if self.h[reaction] < 0:
            raise ValueError("h value for" + reaction + "must not be negative.")
        
        # Reaction possibilities times reaction constants
        return self.rConstants[reaction]*self.h[reaction]

    def calc_a0(self):
        """Calculates some extent of reactibility for the reaction system."""

        if not isinstance(self.a, dict):
            raise TypeError("a must be of type dictionary. Type: " + type(self.a))

        ret = 0
        # sum up all 'a' values
        for val in self.a.values():
            if not isinstance(val, int) and not isinstance(val, float) and not isinstance(val, np.int64):
                raise ValueError("Values in 'a' have to be numbers.")
            elif val < 0:
                raise ValueError("Values in 'a' have to be positive.")
            else:
                ret += val

        return ret

    def advance_time(self):
        """Calculate the next time interval."""

        if self.a0 < 0:
            raise ValueError("a0 can't be negative.")
        if self.times[-1] < 0:
            raise ValueError("Time stamp can't be negative.")
        
        random_number = np.random.random()
        dtime = (1/self.a0)*np.log(1/random_number)  # np.log(x) is for base e, np.log10(x) would be for base 10
        self.times.append(self.times[-1] + dtime)
    
    def _reaction_evaluation(self, key_list, limit):
        """Calculate next reaction to happen."""

        # This method should only be called from choose_next_reaction() so the following Errors become obsolete.
        # But better be safe than sorry.
        # if key_list contains more keys then self.a
        diff1 = [key for key in key_list if key not in self.a.keys()]
        if any(diff1):
            raise KeyError("Could not match the key(s) " + str(diff1) + " to dictionary a.")
        # if key_list contains less keys then self.a
        diff2 = [key for key in self.a.keys() if key not in key_list]
        if any(diff2):
            raise KeyError("Missing key " + str(diff2) + " of dictionary a in key_list.")
        if self.a0 < 0:
            raise ValueError("a0 can't be negative.")

        little_sum = 0
        big_sum = 0
        next_react = ""

        # sum from reaction 1 to reaction M
        for reaction in key_list:
            # upper sum
            big_sum += self.a[reaction]
            # check if random2*a0 = limit is between sums
            # 
            # mu-1               mu
            # SUM  ai < r2*a0 <= SUM ai
            # i=1                i=1
            if little_sum < limit and limit <= big_sum:
                next_react = reaction
                break
            # lower sum
            little_sum += self.a[reaction]

        return next_react

    def choose_next_reaction(self):
        """Preprocessing the reactions for _reaction_evaluation()."""

        # entries of self.a should not be negative
        for entry in self.a.values():
            if entry < 0:
                raise ValueError("Values in 'a' must not be negative.")
        # checking that a0 is not negative
        if self.a0 < 0:
            raise ValueError("a0 can't be negative.")
        # calculate the reaction mu to happen using sums over 'a'i
        random_number = np.random.random()
        limit = random_number*self.a0

        # make sure that all reaction names are always in the same order
        reaction_names = self.a.keys()
        sorted_reactions = sorted(reaction_names)
        next_react = self._reaction_evaluation(sorted_reactions, limit)

        return next_react

    def trace_molecule_changes(self, reaction):

        # check if reaction is a key of the input stoichiometric matrices
        if not reaction in self.L.keys():
            raise KeyError("Reaction should be a key for stoichiometric matrices, respectively the name of the corresponding reaction.")
        # add the changed number of analytes, due to reaction mu, to quantities
        for species in self.quantities.keys():
            # add the value from N to the current number of molecules of one species
            new_quantity = self.quantities[species][-1] + self.N[reaction][species]
            self.quantities[species].append(new_quantity)

    def plot(self, colours=None, outfile="temp_plot"):

        if colours != None:
            if not self.quantities.keys() == colours.keys():
                raise KeyError("Can't match colours.keys() to species.")
            for species in self.quantities.keys():
                plt.plot(self.times, self.quantities[species], colours[species], label=species)
        else:
            for species in self.quantities.keys():
                plt.plot(self.times, self.quantities[species], label=species)
        
        plt.xlabel("time")
        plt.ylabel("total amount")
        plt.legend()
        plt.show()

        if isinstance(outfile, str):
            plt.savefig(outfile)


def monte_carlo_gillespie(rateConstants, L, N, startQuantities, runs=50, time_max=None, reaction_limit=None):
    """Simulate multiple Gillespie runs."""

    if reaction_limit == None and time_max == None:
        raise ValueError("'time_max' and 'reaction_limit' are None. One of them has to limit the simulation.")
    if reaction_limit != None and time_max != None:
        raise ValueError("'time_max' and 'reaction_limit' both given, select one to limit the simulation.")

    mc_Gillespies = []
    # run as many Gillespies as specified above
    for _ in range(runs):
        gillespie_obj = Gillespie(L, N, rateConstants, startQuantities)
        if time_max != None:
            gillespie_obj.run_time_sec(time_max)
        else:
            gillespie_obj.run_n_reactions(reaction_limit)
        mc_Gillespies.append(gillespie_obj)
    
    return mc_Gillespies

def multiplot(gillespies, x_size=None, y_size=None, colours=None, outfile="temp_multiplot"):
    """Plot an amount of Gillespies in subplots."""

    if len(gillespies) == 0:
        raise ValueError("'gillespies' must not be empty.")
    if y_size == None:
        y_size = int(np.ceil(np.sqrt(len(gillespies))))
        # arbitrary, only 3 plots side by side
        if y_size > 3:
            y_size = 3
    elif not isinstance(y_size, int):
        raise ValueError("'y_size' has to be of type int.")
    if x_size == None:
        x_size = int(np.ceil(len(gillespies)/y_size))
    elif not isinstance(x_size, int):
        raise ValueError("'x_size' has to be of type int.")
    
    # create a figure and associated axes
    figure, axs = plt.subplots(x_size, y_size, sharex="col", sharey="row", gridspec_kw={"hspace": 0, "wspace": 0}, figsize=(10,10))
    figure.suptitle("Gillespie: " + str([key+": rC="+str(gillespies[0].rConstants[key]) for key in gillespies[0].rConstants.keys()]))

    # plot the Gillespie runs as long as there are axes and Gillespie objects
    for i, ax in enumerate(axs.flat):
        if i >= len(gillespies):
            break
        current_gil = gillespies[i]
        if colours == None:
            for species in current_gil.quantities.keys():
                ax.plot(current_gil.times, current_gil.quantities[species], label=species)
        else:
            for species in current_gil.quantities.keys():
                ax.plot(current_gil.times, current_gil.quantities[species], colours[species], label=species)
        
        ax.set(xlabel='time [rel]', ylabel='total amount')
        ax.label_outer()

    # create the legend beneath the plots
    plt.legend(loc=(1.1, 0.1), mode="expand", borderaxespad=0.)

    if isinstance(outfile, str):
        plt.savefig(outfile)

def analyte_plot(gillespies, x_size=None, y_size=None, colours=None, outfile="temp_analyte_plot"):
    """Plot the single analytes in distinct diagrams."""

    if len(gillespies) == 0:
        raise ValueError("'gillespies' must not be empty.")
    if y_size == None:
        y_size = int(np.ceil(np.sqrt(len(gillespies[0].quantities))))
        # arbitrary, only 3 plots side by side
        if y_size > 3:
            y_size = 3
    elif not isinstance(y_size, int):
        raise ValueError("'y_size' has to be of type int.")
    if x_size == None:
        x_size = int(np.ceil(len(gillespies[0].quantities)/y_size))
    elif not isinstance(x_size, int):
        raise ValueError("'x_size' has to be of type int.")
    
    figure, axs = plt.subplots(x_size, y_size, gridspec_kw={"hspace": 0.6, "wspace": 0.3}, figsize=(10,10))
    figure.suptitle("Gillespie: " + str([key+": rC="+str(gillespies[0].rConstants[key]) for key in gillespies[0].rConstants.keys()]))

    # plot for every analyte species one diagram
    for i, species in enumerate(gillespies[0].quantities):
        ax = axs.flat[i]
        # plot for all Gillespie runs the graph for the analyte in the current diagram
        for gilles in gillespies:
            if colours == None:
                ax.plot(gilles.times, gilles.quantities[species], label = species)
            else:
                ax.plot(gilles.times, gilles.quantities[species], colours[species], label = species)
        ax.set(xlabel='time [rel]', ylabel='total amount')
        ax.set_title(species)

    if isinstance(outfile, str):
        plt.savefig(outfile)

def make_output_signal(list_gillespies, output_species_names):
    """Calculate the additive signal of the given analytes."""

    # quantities_per_run should be a single list of lists (one for each run) containing the quantities of one analyte
    if not isinstance(list_gillespies, list):
        raise TypeError("Input for make_output_signal() 'list_gillespies' has to be a lsit of Gillespie objects.")
    if not isinstance(output_species_names, list):
        raise TypeError("Input for make_output_signal() 'output_species_names' has to be a list of analyte names.")

    output_signal = []
    for gilles in list_gillespies:
        output_signal_per_run =[]
        # for every date in this Gillespie run
        # all quantities in a run are of the same length
        for i in range(len(gilles.quantities[output_species_names[0]])):
            # sum up all quantities specified by 'output_spec...' regarding their timestamp i to get the output signal at i
            sum_at_timestamp_i = np.sum([gilles.quantities[species][i] for species in output_species_names])
            #for species in output_species_names:
            #    sum_at_time += gilles.quantities[species][i]
            output_signal_per_run.append(sum_at_timestamp_i)
        output_signal.append(output_signal_per_run)

    return output_signal

def get_trimming_time(list_gillespies, window_length=10, step_width=None, vct=0.05):
    """Scans the Gillespie runs for their variances via shifting window and returns the time when variance change threshold is reached."""

    if vct > 1 or vct < 0:
        raise ValueError("Variance change threshold 'vct' has to be a float from zero to one.")
    if not isinstance(vct, float):
        raise ValueError("Variance change threshold 'vct' has to be a float from zero to one.")

    def make_time_quantity_tuples(list_times, list_quantities):
        """Build a list of (time, quantity) tuples."""

        if not len(list_times) == len(list_quantities):
            raise RuntimeError("The amount of times does not equal the amount of quantity data.")

        tuples = []
        # for all times create a (time, quantity) tuple
        for i in range(len(list_times)):
            time_quan = (list_times[i], list_quantities[i])
            tuples.append(time_quan)

        return tuples

    def get_snipping_time(list_tuple_data, window_size=10, step=None, var_cutoff=0.5):
        """Return the time where steady state begins approximately."""

        if step == None:
            step = int(window_size/10)
            if step < 1:
                step = 1
        if not isinstance(step, int):
            raise ValueError("Parameter 'step' has to be an integer, not" + type(step))

        # initialise with highest possible variance and lowest possible position
        current_variance = np.inf
        snipping_position = 0
        # window shifting over all data to determine the local variance
        for i in range(0, len(list_tuple_data) - window_size + 1, step):
            current_window = list_tuple_data[i:i+window_size]
            # variance over analyte quantity
            new_variance = np.var([date[1] for date in current_window])
            # if variance change between the windows is less than 2 percent
            if abs(new_variance - current_variance) < var_cutoff:  # 2 percent is an arbitrary value yet
                snipping_position = i
                break
            
            current_variance = new_variance

        # return the time stamp from which the variance doesn't change dramatically
        return list_tuple_data[snipping_position][0]

    # create (time, quantity) tuples
    all_tuples = []
    possible_snipping_times = []
    for gillespie in list_gillespies:
        snipping_times_per_run = []
        tuples_per_run = {}
        for species in gillespie.quantities.keys():
            time_quantity_tuples = make_time_quantity_tuples(gillespie.times, gillespie.quantities[species])
            time_quantity_tuples.sort()
            tuples_per_run.update({species: time_quantity_tuples})
            # sort the tuples according to their time and decide where to place the begin of steady state
            snipping_time_per_species = get_snipping_time(time_quantity_tuples, var_cutoff=vct, window_size=window_length, step=step_width)
            snipping_times_per_run.append(snipping_time_per_species)
        # calculate for each Gillespie run its own mean settling time
        possible_snipping_times.append(np.mean(snipping_times_per_run))
        all_tuples.append(tuples_per_run)
    # from all runs take the latest settling time to trim the data
    settling_time = np.max(possible_snipping_times)

    return settling_time

def calc_mean_quantity(inp):
    return np.mean([np.mean(date) for date in inp])

def calc_mean_variance(inp):
    return np.mean([np.var(date) for date in inp])

def calc_mean_standard_deviation(inp):
    return np.mean([np.std(date) for date in inp])

def calc_SNR(signal_mean, background_mean):

    if not isinstance(signal_mean, (int, float)):
        raise TypeError("Signal mean has to be a number.")
    if not isinstance(background_mean, (int, float)):
        raise TypeError("Background mean has to be a number.")
    return signal_mean/background_mean

def calc_Variationskoeffizient():
    pass






# # initialise constants
# L = DataFrame({"reaction1":[1, 1, 0, 0, 0],
#             "reaction2":[0, 1, 1, 0, 0],
#             "reaction3":[0, 0, 0, 1, 0],
#             "reaction4":[0, 0, 0, 0, 1]}, index = ["target","probe","interferer","tp-complex","ip-complex"])
# R = DataFrame({"reaction1":[0, 0, 0, 1, 0],
#             "reaction2":[0, 0, 0, 0, 1],
#             "reaction3":[1, 1, 0, 0, 0],
#             "reaction4":[0, 1, 1, 0, 0]}, index = ["target","probe","interferer","tp-complex","ip-complex"])
# N = R - L

# rConstants = dict()
# rConstants["reaction1"] = 0.1
# rConstants["reaction2"] = 0.05
# rConstants["reaction3"] = 0.04
# rConstants["reaction4"] = 0.03

# quantities = [("target", [50]), ("probe", [30]), ("interferer", [10]), ("tp-complex", [5]), ("ip-complex", [5])]

# gillespie_obj = Gillespie(L, N, rConstants, quantities)
# gillespie_obj.run_n_reactions(10)