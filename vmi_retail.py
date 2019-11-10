import numpy as np
from scipy.optimize import minimize


## 1
m = 3
g = 4
eA = np.zeros((m,g,g))    # m*g*g


ea = np.ones((m,g,m,g))    # m*g*m*g
ep = np.ones((m,g,m,g))    # m*g*m*g
K = np.ones((m,g))      # m*g
v = np.ones((m,g,m,g))      # m*g*m*g
u = np.ones((m,g,g))      # m*g*g
beta = np.full((m,g,m,g),0)    # m*g*m*g

beta[0][0] = np.array([[-19,2.2,2.1,1.2],[2.5,1.6,1.2,0],[2.1,0,0,0.8]])
beta[0][1] = np.array([[1.8,-25,1.9,1.5],[0.9,2.2,1.1,0],[1.2,0,0,0.7]])
beta[0][2] = np.array([[1.5,1.9,-22,2],[1,1,21,0],[1.1,0,0,1.6]])
beta[0][3] = np.array([[1.3,18,2.5,-26],[0.7,1,18,0],[0.6,0,0,2.1]])
beta[1][0] = np.array([[2.4,1.6,1,0.7],[-19,1.9,1.3,0],[1,0,0,0.6]])
beta[1][1] = np.array([[1,2.1,1.2,0.8],[1.6,1.8,1.6,0],[1.1,0,0,0.5]])
beta[1][2] = np.array([[0.8,1.1,1.8,1],[15,25,-25,0],[0.6,0,0,0.7]])
beta[2][0] = np.array([[2.3,1.8,1.1,0.8],[1.1,1,0.7,0],[-18,0,0,1.9]])
beta[2][3] = np.array([[1,1.5,2.1,2.2],[0.5,0.6,0.7,0],[1.1,0,0,-23]])

## 2
pw0 = np.ones((g))  # g
## 3
w = np.ones((m))      # m
rho = np.ones((g))
## 4

## 5

A = np.ones((g))
p = np.ones((m,g))      # m*g
# p_m = cp.Variable((g))      # m*g

teta = np.ones((m,g))       # m*g
a = np.ones((m,g))
# a_m = cp.Variable((g))

def calc_uAeA(u,A,eA,m_retail):
    result = np.zeros(g)
    for i in range(g):
        for j in range(g):
            result[i] += u[m_retail][i][j]*np.power(A[j],eA[m_retail][i][j])
    return result

def calc_betapep(beta,p,ep,m_retail):
    result = np.zeros(g)
    for i in range(g):
        for j in range(m):
            if j != m_retail:
                for k in range(g):
                    result[i] += beta[m_retail][i][j][k]*np.power(p[j][k],ep[m_retail][i][j][k])
    return result

def calc_betapep_m(beta,p_m,ep,m_retail):
    result = []
    for i in range(g):
        result.append(0)
    for i in range(g):
        for k in range(g):
            result[i] += beta[m_retail][i][m_retail][k]*np.power(p_m[k],ep[m_retail][i][m_retail][k])
    return result

def calc_vaea(v,a,ea,m_retail):
    result = np.zeros(g)
    for i in range(g):
        for j in range(m):
            if j != m_retail:
                for k in range(g):
                    result[i] += v[m_retail][i][j][k]*np.power(a[j][k],ea[m_retail][i][j][k])
    return result

def calc_vaea_m(v,a_m,ea,m_retail):
    result = []
    for i in range(g):
        result.append(0)
    for i in range(g):
        for k in range(g):
            result[i] += v[m_retail][i][m_retail][k]*np.power(a_m[k],ea[m_retail][i][m_retail][k])
    return result


x0 = np.zeros((2*g))      # m*g
def cacl_DP(p_m,a_m,m_retail):
    uAeA = calc_uAeA(u, A, eA, m_retail)
    betapep = calc_betapep(beta, p, ep, m_retail)
    betapep_m = calc_betapep_m(beta, p_m, ep, m_retail)

    vaea = calc_vaea(v, a, ea, m_retail)
    vaea_m = calc_vaea_m(v, a_m, ea, m_retail)

    DP_0 = K[m_retail] + uAeA + betapep + vaea  # g
    DP = DP_0 + betapep_m + vaea_m
    return DP

def objective(x):
    m_retail = 1

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
    return 100000 - sum(cacl_DP(p_m,a_m,1))
def constraint2(x):
    return 100 - (x[4]+x[5]+x[6]+x[7])
# print(objective(x0))

print('Initial Objective: ' + str(objective(x0)))

b = (0.0,100000,0)
bnds = (b, b, b, b,b,b,b,b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
cons = ([con1,con2])
solution = minimize(objective,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)
x = solution.x


print(x)
print(objective(x))