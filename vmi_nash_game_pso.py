import numpy as np
from scipy.optimize import minimize
import random
from vmi_parameter import *
from vmi_retail import objective,constraint2,cacl_DP
from vmi_manu_pso import *

a = np.zeros((m,g))
A = np.zeros((g))
p = np.zeros((m,g))      # m*g
rho = np.zeros((g))


DA = np.zeros((j,k))                     # j*k
DP = np.zeros((m,g))            # m*g
pw = pw0

c = np.zeros((m))                       # m
crm = np.zeros((s))                     # s
drm = np.zeros((s,l))                     # sl
fpp = np.zeros((g))                      # g
fpm = np.zeros((j))                     # j
fpa = np.zeros((j,k))                     # j*k

if __name__ == "__main__":

    b = (1.0, 1000, 0)
    bnds = (b, b, b, b, b, b, b, b)
    N = 1000
    W = 0.5
    c1 = 0.8
    c2 = 0.9
    population = InitPopulation(N)
    search_space = Space(population, N, Objective)
    for _ in range(100):
        for _m in range(m):
            def constraint1(x):
                p_m = np.array([x[0], x[1], x[2], x[3]])
                a_m = np.array([x[4], x[5], x[6], x[7]])
                return 1000 - sum(cacl_DP(p_m, a_m, 1))*c[_m]


            con1 = {'type': 'ineq', 'fun': constraint1}
            con2 = {'type': 'ineq', 'fun': constraint2}
            cons = ([con1, con2])
            a_m = a[_m,:]
            p_m = p[_m,:]
            # print(a_m)
            # print(p_m)
            x = np.concatenate((p_m,a_m),axis=0)
            solution = minimize(objective, x,args=(_m), method='SLSQP', \
                                bounds=bnds, constraints=cons)
            result = solution.x
            p_m = np.array([result[0], result[1], result[2], result[3]])
            a_m = np.array([result[4], result[5], result[6], result[7]])
            # print(result)
            print(objective(result,_m))
            a[_m,:] = a_m
            p[_m,:] = p_m
            DP[_m] = cacl_DP(p_m, a_m, _m)

        print("-------------------")

        iter = 100

        for i in range(iter):
            search_space.set_pbest()
            search_space.set_gbest()

            search_space.move_particles()

        x = search_space.gvar
        # print(x)
        A = np.array(x[:g])
        c = np.array(x[g:g + m])
        crm = np.array(x[g + m:g + m + s])
        drm = np.reshape(np.array(x[g + m + s:g + m + s + s * l]), (s, l))
        fpp = np.array(x[g + m + s + s * l:g + m + s + s * l + g])
        fpm = np.array(x[g + m + s + s * l + g:g + m + s + s * l + g + j])
        fpa = np.reshape(np.array(x[g + m + s + s * l + g + j:g + m + s + s * l + g + j + j * k]), (j, k))
        rho = np.array(x[g + m + s + s * l + g + j + j * k:])
        print(search_space.obj_value_g)
