import sympy as sp
import pysindy as ps


def set_param_freq(ode_param, freq):
    if ode_param is not None:
        param = [float(x) for x in ode_param.split(',')]
    else:
        param = None

    if freq >= 1:
        freq = int(freq)

    return param, freq


def intercept_library_fun(n_variables):

    if n_variables == 1:
        X0 = sp.symbols('X0')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0), intercept, modules='numpy')
        function_name_intercept = lambda X0: str(intercept)

    elif n_variables == 2:
        X0, X1 = sp.symbols('X0 X1')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1: str(intercept)

    elif n_variables == 3:
        X0, X1, X2 = sp.symbols('X0 X1 X2')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1, X2), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1, X2: str(intercept)

    elif n_variables == 4:
        X0, X1, X2, X3 = sp.symbols('X0 X1 X2 X3')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1, X2, X3), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1, X2, X3: str(intercept)

    else:
        raise ValueError("n_variables must be between 1 and 4")

    intercept_library = ps.CustomLibrary(
        library_functions=[intercept_lambda],
        function_names=[function_name_intercept]
    )

    return intercept_library

