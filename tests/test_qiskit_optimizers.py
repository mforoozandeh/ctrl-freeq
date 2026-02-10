import numpy as np
from qiskit_algorithms.optimizers import OptimizerState
from qiskit_algorithms.optimizers import (
    ADAM,
    COBYLA,
    GSLS,
    L_BFGS_B,
    NELDER_MEAD,
    P_BFGS,
    POWELL,
    SLSQP,
    SPSA,
    TNC,
)
from scipy.optimize import minimize

# Define the Rosenbrock function


def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


# Gradient of the Rosenbrock function
def rosenbrock_grad(x):
    a = 1
    b = 100
    dfdx0 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    dfdx1 = 2 * b * (x[1] - x[0] ** 2)
    return np.array([dfdx0, dfdx1])


# Initial guess
x0 = [0, 0]

# Use scipy's minimize with the BFGS method
result = minimize(rosenbrock, x0=x0, jac=rosenbrock_grad, method="BFGS")

print(result)

# The minimum value is at:
print("Minimum at:", result.x)

state = OptimizerState(
    result.x, result.fun, result.jac, result.nfev, result.njev, result.nit
)
print("Value of the function:", state.fun)


algs = [
    NELDER_MEAD,
    POWELL,
    SLSQP,
    L_BFGS_B,
    TNC,
    P_BFGS,
    SPSA,
    COBYLA,
    GSLS,
    ADAM,
    # AQGD,
    # BOBYQA,
    # SNOBFIT,
    # IMFIL,
    # ISRES,
]


# Define the Rosenbrock function
def rosenbrock(x):
    a = 1
    b = 100
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


# Gradient of the Rosenbrock function
def rosenbrock_grad(x):
    a = 1
    b = 100
    dfdx0 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    dfdx1 = 2 * b * (x[1] - x[0] ** 2)
    return np.array([dfdx0, dfdx1])


# Initial guess
x0 = [0.0, 0.0]


for alg in algs:
    try:
        # Use Qiskit's COBYLA optimizer
        optimizer = alg()
        result = optimizer.minimize(fun=rosenbrock, x0=x0)

        print(f"Results for {alg.__name__} is:\n{result}\n")
    except Exception as e:  # This will catch any error that occurs
        print(f"An error occurred with {alg.__name__}:\n{str(e)}\n")
        continue  # This is optional as the loop will continue to the next item anyway
