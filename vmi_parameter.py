import numpy as np

m = 3
g = 4
j=2
k=3
s=3
l=3
eps = 1e-5

eA = np.zeros((m,g,g))    # m*g*g
eA = np.full((m,g,g),eps)    # m*g*g
eA[0][0] = np.array([0.35,0.26,0.2,0.18])
eA[0][1] = np.array([0.21,0.36,0.21,0.15])
eA[0][2] = np.array([0.17,0.22,0.34,0.21])
eA[0][3] = np.array([0.1,0.19,0.21,0.36])
eA[1][0] = np.array([0.32,0.25,0.12,0])
eA[1][1] = np.array([0.2,0.33,0.2,0])
eA[1][2] = np.array([0.17,0.23, 0.35,0])
# eA[1][3] = np.array([])
eA[2][0] = np.array([0.32,0,0,0.2])
# eA[2][1] = np.array([])
# eA[2][2] = np.array([])
eA[2][3] = np.array([0.23,0,0,0.36])

ea = np.zeros((m,g,m,g))    # m*g*m*g
ea = np.full((m,g,m,g),eps)    # m*g*m*g
ea[0][0] = np.array([[0.39,0.24,0.22,0.2],[0.2,0.16,0.17,0],[0.2,0,0,0.16]])
ea[0][1] = np.array([[0.23,0.46,0.22,0.18],[0.16,0.2,0.16,0],[0.16,0,0,0.16]])
ea[0][2] = np.array([[0.2,0.2,0.37,0.2],[0.18,0.18,0.2,0],[0.18,0,0,0.18]])
ea[0][3] = np.array([[0.19,0.2,0.23,0.42],[0.2,0.16,0.17,0],[0.6,0,0,0.22]])
ea[1][0] = np.array([[0.36,0.3,0.29,0.18],[0.4,0.33,0.3,0],[0.2,0,0,0.16]])
ea[1][1] = np.array([[0.34,0.36,0.3,0.24],[0.4,0.41,0.32,0],[0.16,0,0,0.16]])
ea[1][2] = np.array([[0.24,0.32,0.29,0.28],[0.32,0.34,0.37,0],[0.16,0,0,0.2]])
ea[2][0] = np.array([[0.36,0.31,0.3,0.26],[0.2,0.16,0.14,0],[0.4,0,0,0.29]])
ea[2][3] = np.array([[0.2,0.32,0.38,0.42],[0.13,0.18,0.24,0],[0.2,0,0,0.43]])


ep = np.ones((m,g,m,g))    # m*g*m*g        # =1


K = np.zeros((m,g))      # m*g
K = np.array([[700,800,800,900],[700,900,700,0],[900,0,0,800]])

v = np.zeros((m,g,m,g))      # m*g*m*g
v = np.full((m,g,m,g),eps)      # m*g*m*g
v[0][0] = np.array([[6.3,-2.1,-1.5,-0.6],[-1.9,-1.6,-1,0],[-1.2,0,0,-0.5]])
v[0][1] = np.array([[-1.6,6.6,-1.6,-1],[-1,-1.6,-1,0],[-1,0,0,-1]])
v[0][2] = np.array([[-1.6,-1.6,6.3,-1.6],[-1,-1,-1.6,0],[-0.6,0,0,-1]])
v[0][3] = np.array([[-1.2,-1.5,-1.9,7.2],[-0.6,-1,-1.1,0],[-0.6,0,0,-1.5]])
v[1][0] = np.array([[-2.6,-1.6,-1,-0.6],[6.1,-1.9,-1.3,0],[-1,0,0,-0.7]])
v[1][1] = np.array([[-1,-1.6,-1,-0.6],[-1.6,-6.3,-1.6,0],[-0.6,0,0,-0.6]])
v[1][2] = np.array([[-1,-1.6,-1,-0.6],[-1.6,-6.3,-1.6,0],[-0.6,0,0,-0.6]])
v[2][0] = np.array([[-1.6,-1.2,-1,-0.9],[-1.2,-1,-0.7,0],[8.3,0,0,-1.5]])
v[2][3] = np.array([[-0.7,-0.8,-1,-1.5],[-0.5,-0.6,-1,0],[-1,0,0,7.2]])


u = np.zeros((m,g,g))      # m*g*g
u = np.full((m,g,g),eps)      # m*g*g
u[0][0] = np.array([12,-1.8,-1.5,-1.1])
u[0][1] = np.array([-1.7,18,-1.2,-0.9])
u[0][2] = np.array([-1,-1.9,11,-1.9])
u[0][3] = np.array([-0.9,-1.1,-2.1,16])
u[1][0] = np.array([12,-2.2,-1,0])
u[1][1] = np.array([-1.7,18,-1.8,0])
u[1][2] = np.array([-1.7,-1.9,11,0])
# u[1][3] = np.array([])
u[2][0] = np.array([11,0,0,-1])
# u[2][1] = np.array([])
# u[2][2] = np.array([])
u[2][3] = np.array([-1.3,0,0,18])




beta = np.full((m,g,m,g),eps)    # m*g*m*g
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
pw0 = np.zeros((g))  # g
pw0 = np.array([36,34,32,39])

## 3
w = np.ones((m))      # m    #= 1

# rho = np.zeros((g))     #g
## 4

## 5

# A = np.zeros((g))
# p = np.zeros((m,g))      # m*g
# p_m = cp.Variable((g))      # m*g

teta = np.array([[1.8,1.8,1.4,2.1],[1.4,1.8,1.5,0],[1.4,0,0,2]])       # m*g
# a = np.zeros((m,g))
# a_m = cp.Variable((g))

# w0 = 1                  # 1

## 9
# w = np.one((m))       # m
## 10
TMP = 20000      # 1
## 11
TA = 13350                      # 1

# DP =  np.zeros((m,g))            # m*g
# pw0 = np.zeros((g))

## max
HR = np.zeros((m,g))                      # m*g
HR = np.array([[6.2,5.1,6.1,6.3],[6.2,5.1,6.1,0],[5.5,4.8,0,5.3]])                      # m*g
# teta = np.zeros((m,g))                    # m*g

OP = np.zeros((m))                      # m
OP = np.array([50,40,40])                      # m


HP = np.zeros((g))                      # g
HP = np.array([6.12,4.9,5.8,5.7])


ORM = np.zeros((s))                     # s
ORM = np.array([60,45,95])                     # s

HRM = np.zeros((l))                     # l
HRM = np.array([0.45,0.48,0.52])                   # l


inf = 1e6
TP = np.zeros((m,g))                      # m*g
TP = np.array([[1.4,1.7,1.4,1.4],[1.5,2.5,1.8,inf],[1.4,inf,inf,1.6]])                      # m*g

PCP = np.zeros((g))                     # g
PCP = np.array([1.2,1.3,1.2,1.6])                     # g

PCA = np.zeros((j,k))                     # j*k
PCA = np.array([[0.9,1.1,1.2],[0,0,0]])                     # j*k

PCR = np.zeros((s,l))                      # s*l
PCR = np.array([[0.9,1.1,1.4],[1,inf,1.2],[1.1,1,inf]])                      # s*l

FCP = np.zeros((g))                     # g
FCP = np.array([1800,2000,7000,3100])                     # g


FCM = np.zeros((j))                     # j
FCM = np.array([2000,0])                    # j

FCA = np.array([[1700,1900,4600],[0,0,0]])                     # j*k

# DA = np.zeros((j,k))                     # j*k
