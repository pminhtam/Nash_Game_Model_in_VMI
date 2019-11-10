import cvxpy as cp
import numpy as np
# Create two scalar optimization variables.
x = cp.Variable()
y = cp.Variable()
print(x.value)
# Create two constraints.
constraints = [x + y == 1,
               x - y >= 1]

# Form objective.
# obj = cp.Minimize((x - y)**2)
# obj = cp.Minimize(cp.power(x - y,2)+ x)
obj = cp.Minimize(x*y)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve(gp=True)  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)