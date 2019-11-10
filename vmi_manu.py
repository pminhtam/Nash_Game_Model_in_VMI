import numpy as np
from scipy.optimize import minimize



m = 3
g = 4
j=2
k=3
s=3
l=3
## 1
## 2
## 3
w0 = 1                  # 1

## 9
wm = np.ones((m))       # m
## 10
TMP = 1      # 1
## 11
TA = 1                      # 1

DP =  np.ones((m,g))                      # m*g
## find

A = np.ones((g))                       # g
c = np.ones((m))                       # m
crm = np.ones((s))                     # s
drm = np.ones((s,l))                     # sl
fpp = np.ones((g))                      # g
fpm = np.ones((j))                     # j
fpa = np.ones((j,k))                     # j*k
rho = np.zeros((g))     # g

pw0 = np.ones((g))
pw = pw0 - np.multiply(rho,DP)


## max
HR = np.ones((m,g))                      # m*g
teta = np.ones((m,g))                    # m*g

OP = np.ones((m))                      # m
HP = np.ones((g))                      # g

ORM = np.ones((s))                     # s
HRM = np.ones((l))                     # l

TP = np.ones((m,g))                      # m*g
PCP = np.ones((g))                     # g

PCA = np.ones((j,k))                     # j*k

PCR = np.ones((s,l))                      # s*l

FCP = np.ones((g))                     # g
FCM = np.ones((j))                     # j
FCA = np.ones((j,k))                     # j*k
DA = np.ones((j,k))                     # j*k


NP1 = np.sum(np.multiply(DP,pw))

NP2 = np.sum(np.multiply(c/2,np.sum(np.multiply(DP,HR),axis=1))) - np.sum(np.multiply(teta,DP))
NP3 = np.sum(np.multiply(1/c,OP))+np.sum(np.multiply(c/2,np.sum(np.multiply(DP,HP),axis=1)))
NP4 = np.sum(np.multiply(1/crm,ORM))+np.sum(np.multiply(crm/2,np.sum(np.multiply(drm,HRM),axis=1)))
NP5 = np.sum(np.multiply(DP,TP)) + np.sum(np.multiply(DP,PCP))+np.sum(np.multiply(DA,PCA))+np.sum(np.multiply(drm,PCR))
NP6 = np.sum(np.multiply(fpp,FCP)) + np.sum(np.multiply(fpm,FCM))+np.sum(np.multiply(fpa,FCA))+np.sum(A)

NP = NP1-NP2-NP3-NP4-NP5-NP6
print(NP)