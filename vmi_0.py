import cvxpy as cp
import numpy as np


## 1
m = 3
g = 4
eA = np.ones((m,g,g))    # m*g*g
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
p_m = cp.Variable((g),pos=True)      # m*g

teta = np.ones((m,g))       # m*g
a = np.ones((m,g))
a_m = cp.Variable((g),pos=True)

m_retail = 1
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
            # result[i] += beta[m_retail][i][m_retail][k]*np.power(p_m[k],ep[m_retail][i][m_retail][k])
            result[i] += beta[m_retail][i][m_retail][k]*p_m[k]
    return result

def calc_vaea(v,a,ea,m_retail):
    result = np.zeros(g)
    for i in range(g):
        for j in range(m):
            if j != m_retail:
                for k in range(g):
                    result[i] += v[m_retail][i][j][k]*np.power(a[j][k],ea[m_retail][i][j][k])
    return result

uAeA = calc_uAeA(u,A,eA,m_retail)
betapep = calc_betapep(beta,p,ep,m_retail)
betapep_m = calc_betapep_m(beta,p_m,ep,m_retail)

vaea = calc_vaea(v,a,ea,m_retail)


DP_0 = K[m_retail] + uAeA + betapep + vaea    # g
DP = []
for i in range(len(DP_0)):
    DP.append(DP_0[i] + betapep_m[i])
    # DP.append(DP_0[i])
print("89:   ",DP[0])
# pw = pw0 - cp.multiply(rho,DP)       # g
rhoDP = []       # g
for i in range(len(rho)):
    rhoDP.append(rho[i]*DP[i])

pw = []
for i in range(len(pw0)):
    pw.append(pw0[i] + rhoDP[i])

print("103:   ",pw)


NP1 = []
for i in range(len(DP)):
    NP1.append(DP[i] * p_m[i])

NP2 = []
for i in range(len(DP)):
    NP2.append(DP[i] * pw[i])

NP3 = []
for i in range(len(DP)):
    NP3.append(DP[i] * teta[m_retail][i])

NP = cp.sum(NP1) - cp.sum(NP2) - cp.sum(NP3) - cp.sum(a_m)
# NP = cp.sum(teta*DP)
print("120:   ",NP)
print("121:   ",NP.log_log_curvature)
constraints = [cp.sum(a_m) <= 100]


obj = cp.Maximize(NP)

prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print(p_m.value)
print(a_m.value)