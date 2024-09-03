import sys
import os
import pytest
import pandas as pd
import numpy as np
import pytest
import model_solver as ms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Negative Test: Invalid Equations

def test_invalid_equation_format():
    equations = ['x1 == a1']  # Invalid syntax with '=='
    endogenous = ['x1']
    with pytest.raises(SyntaxError):  # Assuming a SyntaxError or custom error is raised
        ms.ModelSolver(equations, endogenous)

def test_unsupported_operation():
    equations = ['x1 = sqrt(a1)']  # Assuming 'sqrt' is not supported
    endogenous = ['x1']
    with pytest.raises(ValueError):  # Assuming a ValueError or custom error is raised
        ms.ModelSolver(equations, endogenous)

def test_mismatched_equation_and_endogenous_count():
    equations = ['x1 = a1', 'x2 = a2']
    endogenous = ['x1']  # Mismatch in number of endogenous variables
    with pytest.raises(ValueError):  # Assuming a ValueError or custom error is raised
        ms.ModelSolver(equations, endogenous)

# Negative Test: Invalid Endogenous Variables

def test_nonexistent_endogenous_variable():
    equations = ['x1 = a1']
    endogenous = ['x2']  # 'x2' is not in any equation
    with pytest.raises(RuntimeError):  # Assuming a RuntimeError or custom error is raised
        ms.ModelSolver(equations, endogenous)

def test_duplicate_endogenous_variable():
    equations = ['x1 = a1', 'x1 = a2']  # Duplicate equation for 'x1'
    endogenous = ['x1', 'x1']  # Duplicate endogenous variable
    with pytest.raises(RuntimeError):  # Assuming a RuntimeError or custom error is raised
        ms.ModelSolver(equations, endogenous)

# Negative Test: Invalid Input Data

def test_non_numeric_input_data():
    equations = ['x1 = a1']
    endogenous = ['x1']
    input_data = pd.DataFrame({'x1': ['a', 'b', 'c'], 'a1': [1, 2, 3]})  # Non-numeric input data
    model = ms.ModelSolver(equations, endogenous)
    with pytest.raises(TypeError):  # Assuming a TypeError or custom error is raised
        model.solve_model(input_data)

def test_missing_column_in_input_data():
    equations = ['x1 = a1']
    endogenous = ['x1']
    input_data = pd.DataFrame({'a1': [1, 2, 3]})  # Missing column 'x1'
    model = ms.ModelSolver(equations, endogenous)
    with pytest.raises(KeyError):  # Assuming a KeyError or custom error is raised
        model.solve_model(input_data)

def test_out_of_range_period_index():
    equations = ['x1 = a1']
    endogenous = ['x1']
    input_data = pd.DataFrame({'x1': [1, 2, 3], 'a1': [4, 5, 6]})
    model = ms.ModelSolver(equations, endogenous)
    model.solve_model(input_data)
    with pytest.raises(IndexError):  # Assuming an IndexError is raised
        model.trace_to_exog_vals(1, period_index=100)  # Invalid period index

# Negative Test: Invalid Operations

def test_invalid_switch_endo_vars():
    equations = ['x1 = a1']
    endogenous = ['x1']
    model = ms.ModelSolver(equations, endogenous)
    with pytest.raises(RuntimeError):  # Assuming a RuntimeError or custom error is raised
        model.switch_endo_vars(['nonexistent_var'], ['a1'])  # Switching a nonexistent endogenous variable

def test_invalid_trace_block():
    equations = ['x1 = a1']
    endogenous = ['x1']
    model = ms.ModelSolver(equations, endogenous)
    with pytest.raises(IndexError):  # Assuming an IndexError or custom error is raised
        model.trace_to_exog_vars(999)  # Invalid block number
