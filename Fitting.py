import numpy as np
import pandas as pd
from Gillespie import Gillespie

def divide_time_axis_equidistantly(list_of_times):
    """Divides a list of numbers pairwise in the middle. Returns a list of tuples of sections."""
    # only a list for input
    if not isinstance(list_of_times, list):
        raise ValueError('The input list has to be a list of times.')
    # no empty list
    elif list_of_times == []:
        raise ValueError('The input list must not be empty.')
    # only lists with two and more items
    elif len(list_of_times) == 1:
        raise ValueError('The input list has to contain more than one entry.')
    # list items shall be of type integer or float
    for list_item in list_of_times:
        if not (isinstance(list_item, int) or isinstance(list_item, float)):
            raise ValueError('The list items shall be of type integer or float.')
    
    time_ranges = []
    for i, time in enumerate(list_of_times):
        # if last item in list, there is only one intermediate
        if i == len(list_of_times)-1:
            time_ranges.append((old_intermediate, time))
            break
        # calculate the next intermediate
        intermediate = (list_of_times[i+1]+time)/2
        # if it is the first item, there is only one intermediate
        if i == 0:
            time_ranges.append((time, intermediate))
        # all other time ranges range from intermediate to intermediate
        else:
            time_ranges.append((old_intermediate, intermediate))
        old_intermediate = intermediate
    return time_ranges

# TODO: Tests schreiben
def assign_simulation_times_to_time_ranges(time_ranges, gillespies):
    """Takes a list of gillespie simulations and a list of time intervals and matches the time stamps of each gillespie simulation to an interval."""
    rows = []
    times_for_all_gillespie_objects = []
    # for each gillespie object
    for i, gilles in enumerate(gillespies):
        rows.append(gilles)
        times_for_all_gillespie_objects.append([])
        # for each interval in the given list
        for time_range in time_ranges:
            times_in_time_range = []
            for time in gilles.times:
                # check if a time from the gillespie simulation matches the current interval
                if (time >= time_range[0] and time < time_range[1]) or (time < time_range[0] and time >= time_range[1]):
                    times_in_time_range.append(time)
            # add all matched times to the current gillespie objects list
            times_for_all_gillespie_objects[i].append(times_in_time_range)

    all_times = pd.DataFrame(times_for_all_gillespie_objects, columns=time_ranges, index=rows)
    return all_times
# TODO: Mittelwert der Simulationsdaten im Messbereich bestimmen
# TODO: Tests schreiben
def arithmetic_means(dataframe):
    # TODO: Beschreibung ausdenken und einfügen
    """Beschreibung..."""
    ret = pd.DataFrame(np.zeros(dataframe.shape), columns=dataframe.columns, index=dataframe.index)
    for col, ind in (dataframe.columns, dataframe.index):
        # each entry of 'dataframe' should contain a list of values
        # calculate the arithmetic mean and save it to the corresponding location in 'ret'
        mean = np.mean(dataframe[col][ind])
        ret[col][ind]=mean
    
    return ret
# TODO: Tests schreiben
def evaluation(df_measured, df_simulated_means):
    """Evaluates how much a simulations differs from measured data."""
    iterated_sum = 0
    absolutes_sum = 0
    for col, ind in (df_measured.columns, df_measured.index):
        difference = df_measured[col][ind]-df_simulated_means[col][ind]
        iterated_sum += difference
        absolutes_sum += abs(difference)
    
    return (iterated_sum, absolutes_sum)

def fitness(parameters):
    # TODO: Ist es sinnvoller eine Trajektrie oder gleich mehrere zu erzeugen, um zu fitten?
    trajectory = Gillespie.monte_carlo_gillespie(parameters, L_lenser, N_lenser, startQuantities_lenser, runs=runs_lenser, reaction_limit=reaction_max_lenser)
    # Mittelwerte bilden
    # Addieren und Normieren (auf feste Anzahl in ROI bspw. 200)
    quality_of_fitness = evaluation()
    return quality_of_fitness


# initialise constants
species_lenser = ["x","y","z"]  # z inner und x free, y loosely bound
# loss side
L_lenser = pd.DataFrame({"reaction01":[0, 0, 0],  # außen - x
            "reaction02":[1, 0, 0],  # x - y
            "reaction03":[0, 1, 0],  # y - z
            "reaction11":[1, 0, 0],  # x - außen
            "reaction12":[0, 1, 0],  # y - x
            "reaction13":[0, 0, 1]},  # z - y
            index = species_lenser)
# gain side
R_lenser = pd.DataFrame({"reaction01":[1, 0, 0],  # außen - x
            "reaction02":[0, 1, 0],  # x - y
            "reaction03":[0, 0, 1],  # y - z
            "reaction11":[0, 0, 0],  # x - außen
            "reaction12":[1, 0, 0],  # y - x
            "reaction13":[0, 1, 0]},  # z - y
            index = species_lenser)
N_lenser = R_lenser - L_lenser

# Do reaction 1 and 2 have nearly the same rate constants?
rateConstants_lenser = {"reaction01": 1,
                  "reaction02": 1,
                  "reaction03": 1,
                  "reaction11": 0.1,
                  "reaction12": 1,
                  "reaction13": 1}

startQuantities_lenser = {"x":[0],"y":[0],"z":[0]}

reaction_max_lenser = 20000
runs_lenser = 2

colours_lenser = {"x":"red", "y":"forestgreen", "z":"lightblue"}

measurement = pd.read_csv("./../Daten/FRAP_comparison_all_new_AssemblyDynamicsOfPMLNuclearBodiesInLivingCells_cleaned.csv")
measured_times = measurement["time[s]"]
time_intervals = divide_time_axis_equidistantly(measured_times)
assigned_simulation_data = assign_simulation_times_to_time_ranges(time_intervals, gillespies=None)
arithmetic_means(assigned_simulation_data)
res = evaluation(measurement, assigned_simulation_data) # Achtung! Es muss bei den "measurement"Daten noch die Indexreihe als Intervall angelegt werden.