import numpy as np
from scipy.optimize import minimize
from vmi_parameter import *



rho = np.zeros((g))
#
A = np.zeros((g))
A = np.array([3787,3562,0,6200])
p = np.zeros((m,g))      # m*g
print(p.shape)
p = np.array([[67.88,61.02,0,55.72],[59.7,69.76,0,0],[70.69,0,0,63.57]])      # m*g
print(p.shape)
a = np.zeros((m,g))
a = np.array([[1049.85,2850.10,0,631.43],[357.29,1681.27,0,0],[2509.17,0,0,1501.64]])
x0 = np.zeros((2*g))      # m*g

product_for_retail = np.array([[1,1,1,1],[1,1,1,0],[1,0,0,1]])
# ep = 1-np.array([[1,1,1,1],[1,1,1,0],[1,0,0,1]])


def calc_uAeA(u,A,eA,m_retail):
    result = np.zeros(g)
    for i in range(g):
        for j in range(g):
            result[i] += product_for_retail[m_retail][i]*product_for_retail[m_retail][j]*u[m_retail][i][j]*np.power(A[j],eA[m_retail][i][j])
    return result

def calc_betapep(beta,p,ep,m_retail):
    result = np.zeros(g)
    for i in range(g):
        for j in range(m):
            if j != m_retail:
                for k in range(g):
                    result[i] += product_for_retail[j][k]*product_for_retail[j][i]*beta[m_retail][i][j][k]*np.power(p[j][k],ep[m_retail][i][j][k])
                    # result[i] += beta[m_retail][i][j][k]*np.power(p[j][k],ep[m_retail][i][j][k])
    return result

def calc_betapep_m(beta,p_m,ep,m_retail):
    result = []
    for i in range(g):
        result.append(0)
    for i in range(g):
        for k in range(g):
            result[i] += product_for_retail[m_retail][k]*product_for_retail[m_retail][i]*beta[m_retail][i][m_retail][k]*np.power(p_m[k],ep[m_retail][i][m_retail][k])
    return result

def calc_vaea(v,a,ea,m_retail):
    result = np.zeros(g)
    for i in range(g):
        for j in range(m):
            if j != m_retail:
                for k in range(g):
                    result[i] += product_for_retail[j][k]*product_for_retail[j][i]*v[m_retail][i][j][k]*np.power(a[j][k],ea[m_retail][i][j][k])
    return result

def calc_vaea_m(v,a_m,ea,m_retail):
    result = []
    for i in range(g):
        result.append(0)
    for i in range(g):
        for k in range(g):
            result[i] += product_for_retail[m_retail][k]*product_for_retail[m_retail][i]*v[m_retail][i][m_retail][k]*np.power(a_m[k],ea[m_retail][i][m_retail][k])
    return result


def cacl_DP(p_m,a_m,m_retail):
    uAeA = calc_uAeA(u, A, eA, m_retail)
    betapep = calc_betapep(beta, p, ep, m_retail)
    betapep_m = calc_betapep_m(beta, p_m, ep, m_retail)

    vaea = calc_vaea(v, a, ea, m_retail)
    vaea_m = calc_vaea_m(v, a_m, ea, m_retail)

    DP_0 = K[m_retail] + uAeA + betapep + vaea  # g
    DP = DP_0 + betapep_m + vaea_m
    return DP

def objective(x,m_retail):
    # m_retail = x[8]

    p_m = np.array([x[0],x[1],x[2],x[3]])
    a_m = np.array([x[4],x[5],x[6],x[7]])

    DP = cacl_DP(p_m,a_m,m_retail)
    # pw = pw0 - cp.multiply(rho,DP)       # g
    rhoDP = np.multiply(rho,DP)       # g

    pw = pw0 - rhoDP

    NP1 = np.multiply(DP,p_m)

    NP2 = np.multiply(DP,pw)

    NP3 = np.multiply(DP,teta)

    NP = np.sum(NP1) - np.sum(NP2) - np.sum(NP3) - np.sum(a_m)
    # NP = cp.sum(teta*DP)
    return -NP
def constraint1(x):
    p_m = np.array([x[0],x[1],x[2],x[3]])
    a_m = np.array([x[4],x[5],x[6],x[7]])
    return 1000 - sum(cacl_DP(p_m,a_m,1))
def constraint2(x):
    return 100 - (x[4]+x[5]+x[6]+x[7])
# print(objective(x0))


if __name__ == "__main__":
    # print('Initial Objective: ' + str(objective(x0,1)))
    # b = (1,1000,0)
    # bnds = (b, b, b, b,b,b,b,b)
    # con1 = {'type': 'ineq', 'fun': constraint1}
    # con2 = {'type': 'ineq', 'fun': constraint2}
    # cons = ([con1,con2])
    # solution = minimize(objective,x0,args=(1),method='SLSQP',\
    #                     bounds=bnds,constraints=cons)
    # x = solution.x
    # print(x)
    # print(objective(x,1))
    p_m = np.array([67.88,61.02,0,55.72])
    a_m = np.array([1049.85,2850.10,0,631.43])

    DP = cacl_DP(p_m, a_m, 0)
    print(DP)