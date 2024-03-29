from context import *
from Gillespie import divide_time_axis_equidistantly
from Gillespie import evaluation
from pandas import DataFrame

import pytest

def test_divide_time_axis_equidistantly():
    # no empty inputs
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly([])
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly(None)
    # no floats
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly(0.3)
    # no strings
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly("AA")
    # no dictionaries
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly({"one":1, "two":1})
    # no tuples
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly((1,2,"three"))
    # one list of number values only with at least two entries
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly([1.0])
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly([5,6,'a'])
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly([1,[2,3],4])
    with pytest.raises(ValueError):
        divide_time_axis_equidistantly([{'key':'item'},2])
    assert divide_time_axis_equidistantly([0.0,5.0,7,10]) == [(0,2.5),(2.5,6),(6,8.5),(8.5,10)]

def test_evaluation():
    df_measured = DataFrame({'PML I WT':[1.0,0.09,0.16,0.21,0.26,0.29,0.32]},index=[(-20,-10),(-10,10),(10,30),(30,50),(50,70),(70,90),(90,100)])
    df_simulation = DataFrame({'PML I WT':[1.0,0.1,0.15,0.2,0.25,0.3,0.35]},index=[(-20,-10),(-10,10),(10,30),(30,50),(50,70),(70,90),(90,100)])
    assert evaluation(df_measured, df_simulation) == (-0.02,0.08)