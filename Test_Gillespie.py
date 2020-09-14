from Gillespie import Gillespie
from pandas import DataFrame

import pytest

def get_gill():
    L = DataFrame({"reaction1":[1, 1, 0, 0, 0],
        "reaction2":[0, 1, 1, 0, 0],
        "reaction3":[0, 0, 0, 1, 0],
        "reaction4":[0, 0, 0, 0, 1]}, index = ["target","probe","interferer","tp-complex","ip-complex"])
    R = DataFrame({"reaction1":[0, 0, 0, 1, 0],
        "reaction2":[0, 0, 0, 0, 1],
        "reaction3":[1, 1, 0, 0, 0],
        "reaction4":[0, 1, 1, 0, 0]}, index = ["target","probe","interferer","tp-complex","ip-complex"])
    N = R - L

    rConstants = dict()
    rConstants["reaction1"] = 0.1
    rConstants["reaction2"] = 0.05
    rConstants["reaction3"] = 0.04
    rConstants["reaction4"] = 0.03

    quantities = dict()
    quantities["target"] = [5000]
    quantities["probe"] = [3000]
    quantities["interferer"] = [1000]
    quantities["tp-complex"] = [0]
    quantities["ip-complex"] = [0]

    ret = Gillespie(L, N, rConstants, quantities)
    return ret

def run_gill(): 
    obj = get_gill()
    obj.run_time_sec(5)
    return obj

class Test_Gillespie():

    def test_run_n_reactions(self):
        gillespie_obj = get_gill()

        # no empty inputs
        with pytest.raises(ValueError):
            gillespie_obj.run_n_reactions("")
        with pytest.raises(ValueError):
            gillespie_obj.run_n_reactions(None)
        # no floats
        with pytest.raises(ValueError):
            gillespie_obj.run_n_reactions(0.3)
        # no strings
        with pytest.raises(ValueError):
            gillespie_obj.run_n_reactions("AA")
        # no negative values
        with pytest.raises(ValueError):
            gillespie_obj.run_n_reactions(-1)
        with pytest.raises(ValueError):
            gillespie_obj.run_n_reactions(-0.3)


    def test_run_time_sec(self):
        gillespie_obj = get_gill()
        # no null pointer
        with pytest.raises(ValueError):
            gillespie_obj.run_time_sec(None)
        # no negative values
        with pytest.raises(ValueError):
            gillespie_obj.run_time_sec(-2)
        # no strings
        with pytest.raises(ValueError):
            gillespie_obj.run_time_sec("")
    
    def test_run_single_step(self):
        gillespie_obj = get_gill()

        # the keys for the reactions should be equal
        set1 = set(gillespie_obj.rConstants.keys())
        set2 = set(gillespie_obj.L.keys())
        assert(set1.difference(set2) == set())
        assert(set2.difference(set2) == set())
        
        for reaction in gillespie_obj.rConstants.keys():
            # the keys for the species should be equal
            set3 = set(gillespie_obj.L[reaction].keys())
            set4 = set(gillespie_obj.quantities.keys())
            assert(set3.difference(set4) == set())
            assert(set4.difference(set3) == set())

    def test_get_reactants(self):
        gillespie_obj = get_gill()
        # with creating a DataFrame, all integers convert to type np.int64
        # no negative amounts of species
        gillespie_obj.L = DataFrame({"reaction1":[1, -1, 0, 0, 0],
                                     "reaction2":[0.1, 0, 0, 0, 0],
                                     "reaction3":[1, 0, 0, 0, 0],
                                     "reaction4":[2, 0, 0, 0, 0],
                                     "reaction5":[0, 1, 1, 0, 0]},
                                     index = ["target","probe","interferer","tp-complex","ip-complex"])
        with pytest.raises(ValueError):
            gillespie_obj.get_reactants("reaction1")
        with pytest.raises(ValueError):
            gillespie_obj.get_reactants("reaction2")
        # strings in DataFrame are handeled by pandas package
        assert(gillespie_obj.get_reactants("reaction3") == {"target":1})
        assert(gillespie_obj.get_reactants("reaction4") == {"target":2})
        assert(gillespie_obj.get_reactants("reaction5") == {"interferer": 1, "probe": 1})

    def test_calc_h(self):
        """WARNING: Depends on calc-reactants()"""

        gillespie_obj = get_gill()
        # a reaction should be given
        with pytest.raises(ValueError):
            gillespie_obj.calc_h(None)
        with pytest.raises(ValueError):
            gillespie_obj.calc_a(5.4)
        # only strings, which are key of the defined dictionaries
        with pytest.raises(KeyError):
            gillespie_obj.calc_h("")
        with pytest.raises(KeyError):
            gillespie_obj.calc_h("marmalade")
        
        gillespie_obj.L = DataFrame({"reaction1":[1, -1, 0, 0, 0],
                                     "reaction2":[0.1, 0, 0, 0, 0],
                                     "reaction3":[1, 0, 0, 0, 0],
                                     "reaction4":[2, 0, 0, 0, 0],
                                     "reaction5":[0, 1, 1, 0, 0]},
                                     index = ["target","probe","interferer","tp-complex","ip-complex"])
        gillespie_obj.quantities["target"] = [40, 10]
        gillespie_obj.quantities["probe"] = [3000, 2000, 1000]
        gillespie_obj.quantities["interferer"] = [35, 15]
        # one reactant
        assert(gillespie_obj.calc_h("reaction3") == 10)
        # two reactants
        assert(gillespie_obj.calc_h("reaction4") == 45)
        assert(gillespie_obj.calc_h("reaction5") == 15000)


    def test_calc_a(self):
        gillespie_obj = get_gill()
        # reaction should be a string type key for defined dictionaries
        with pytest.raises(ValueError):
            gillespie_obj.calc_a(5.4)
        with pytest.raises(KeyError):
            gillespie_obj.calc_a("Zitterpappel")
        # no negative h or rConstant values
        with pytest.raises(ValueError):
            gillespie_obj.rConstants["reaction1"] = 0.5
            gillespie_obj.h["reaction1"] = -2
            gillespie_obj.calc_a("reaction1")
        with pytest.raises(ValueError):
            gillespie_obj.rConstants["reaction1"] = -0.5
            gillespie_obj.h["reaction1"] = -2
            gillespie_obj.calc_a("reaction1")
        
        gillespie_obj.rConstants["reaction1"] = 0.5
        gillespie_obj.h["reaction1"] = 15000
        assert(gillespie_obj.calc_a("reaction1") == 7500)

    def test_calc_a0(self):
        gillespie_obj = get_gill()
        # a should be given and of type dictionary
        with pytest.raises(TypeError):
            gillespie_obj.a = None
            gillespie_obj.calc_a0()
        with pytest.raises(TypeError):
            gillespie_obj.a = "Dictionary"
            gillespie_obj.calc_a0()
        # values of a have to be numbers
        with pytest.raises(ValueError):
            gillespie_obj.a = {"reaction1":5.0, "reaction2":8, "reaction3":["list"]}
            gillespie_obj.calc_a0()
        # values of a have to be positive
        with pytest.raises(ValueError):
            gillespie_obj.a = {"reaction1":5.0, "reaction2":-8, "reaction3":13}
            gillespie_obj.calc_a0()
        # shall ever return the sum of all values in a
        gillespie_obj.a = {"reaction1":12, "reaaction2":38.4, "reaction3":20}
        assert(gillespie_obj.calc_a0() == 70.4)

    def test_advance_time(self):
        gillespie_obj = get_gill()
        
        # if the inputs are negative, the time goes backwards
        with pytest.raises(ValueError):
            gillespie_obj.times.append(-0.001)
            gillespie_obj.a0 = 350
            gillespie_obj.advance_time()
        with pytest.raises(ValueError):
            gillespie_obj.times.append(0.5)
            gillespie_obj.a0 = -3.8
            gillespie_obj.advance_time()
        with pytest.raises(ValueError):
            gillespie_obj.times.append(-0.6)
            gillespie_obj.a0 = -20
            gillespie_obj.advance_time()

        current_time = 0.00000013
        gillespie_obj.times[-1] = current_time
        gillespie_obj.a0 = 10000
        # time shall increase, not decrease
        gillespie_obj.advance_time()
        assert(gillespie_obj.times[-1] > current_time)
    
    def test__reaction_evaluation(self):
        gillespie_obj = get_gill()
        
        gillespie_obj.a = {"reaction3": 5000, "reaction4": 3000, "reaction5": 2000}
        gillespie_obj.a0 = 10000
        random_number = 0.51234
        limit = gillespie_obj.a0*random_number
        # check that sorted_keys contains all and only the keys of self.a
        with pytest.raises(KeyError):
            sorted_keys = ["reaction2", "reaction3", "reaction4", "reaction5"]
            gillespie_obj._reaction_evaluation(sorted_keys, limit)
        with pytest.raises(KeyError):
            sorted_keys = ["reaction3", "reaction4"]
            gillespie_obj._reaction_evaluation(sorted_keys, limit)
        # a0 should not be negative
        with pytest.raises(ValueError):
            sorted_keys = ["reaction3", "reaction4", "reaction5"]
            gillespie_obj.a0 = -180.5
            gillespie_obj._reaction_evaluation(sorted_keys, limit)
        
        gillespie_obj.a0 = 10000
        sorted_keys = ["reaction3", "reaction4", "reaction5"]
        assert(gillespie_obj._reaction_evaluation(sorted_keys, limit) == "reaction4")

    def test_choose_next_reaction(self):
        gillespie_obj = get_gill()
        
        # raise an error if a0 is negative
        gillespie_obj.a = {"reaction1":12, "reaction2":38.4, "reaction3":20}
        with pytest.raises(ValueError):
            gillespie_obj.a0 = -1
            gillespie_obj.choose_next_reaction()
        # should return a reaction string s=reaction1, reaction2,...
        gillespie_obj.a0 = 70.4
        assert(type(gillespie_obj.choose_next_reaction()) == str)

    def test_trace_molecule_changes(self):
        gillespie_obj = get_gill()
        
        # the reaction should be a key for the DataFrames
        # no number
        with pytest.raises(KeyError):
            gillespie_obj.trace_molecule_changes(0)
        # no nonesence strings
        with pytest.raises(KeyError):
            gillespie_obj.trace_molecule_changes("alligator")

        # save the current length of the quantities dictionary
        current_lengths_of_quantities = dict()
        for key in gillespie_obj.quantities.keys():
            current_length = len(gillespie_obj.quantities[key])
            current_lengths_of_quantities.update({key: current_length})
        # write molecule changes facing reaction 1
        gillespie_obj.trace_molecule_changes("reaction3")
        for key in gillespie_obj.quantities.keys():
            assert(len(gillespie_obj.quantities[key]) == current_lengths_of_quantities[key]+1)