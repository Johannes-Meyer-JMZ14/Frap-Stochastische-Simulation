import numpy as np
import pandas as pd
from Gillespie import *

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
def assign_simulation_times_to_time_ranges_average(time_intervals, times, list_gillespie_data):
    """Takes a list of time intervals and a list of tuples of [gillespie times] and [gillespie values]
       and matches the time stamps of each gillespie simulation to an interval and averages all values
       in one interval."""
    averaged_data_for_each_run = []
    # for each gillespie object
    for i, gillespie_run in enumerate(list_gillespie_data):
        averaged_data_for_each_run.append([])
        # for each interval in the given list
        for time_range in time_intervals:
            data_in_time_range = []
            for j, time in enumerate(gillespie_run[0]):
                # check if a time from the gillespie simulation matches the current interval
                if (time >= time_range[0] and time < time_range[1]) or (time < time_range[0] and time >= time_range[1]):
                    data_in_time_range.append(gillespie_run[1][j])
            # add all matched times to the current gillespie objects list
            averaged_data_for_each_run[i].append(np.mean(data_in_time_range))

    numpy_array = np.array(averaged_data_for_each_run)
    transposed_array = np.transpose(numpy_array)
    all_time_averages = pd.DataFrame(transposed_array, index=times)
    return all_time_averages
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
    # iterated_sum = 0
    absolutes_sum = 0
    for col, ind in (df_measured.columns, df_measured.index):
        difference = df_measured[col][ind]-df_simulated_means[col][ind]
        # iterated_sum += difference
        absolutes_sum += abs(difference)
    
    return absolutes_sum  # (iterated_sum, absolutes_sum)

#measurement = pd.read_csv("./../Daten/FRAP_comparison_all_new_AssemblyDynamicsOfPMLNuclearBodiesInLivingCells_cleaned.csv")
#measured_times = measurement["time[s]"]
#time_intervals = divide_time_axis_equidistantly(measured_times)
#assigned_simulation_data = assign_simulation_times_to_time_ranges(time_intervals, gillespies=None)
#arithmetic_means(assigned_simulation_data)
#res = evaluation(measurement, assigned_simulation_data) # Achtung! Es muss bei den "measurement"Daten noch die Indexreihe als Intervall angelegt werden.