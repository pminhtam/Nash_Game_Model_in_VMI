import numpy as np
from vmi_parameter import *
import random

DA = np.ones((j,k))                     # j*k
DP = np.ones((m,g))            # m*g
pw = pw0

# print(DP.flatten())
## find

# A = np.ones((g))                       # g
# c = np.ones((m))                       # m
# crm = np.ones((s))                     # s
# drm = np.ones((s,l))                     # sl
# fpp = np.ones((g))                      # g
# fpm = np.ones((j))                     # j
# fpa = np.ones((j,k))                     # j*k
# rho = np.zeros((g))                    # g
#
# NP1 = np.sum(np.multiply(DP, pw))
#
# NP2 = np.sum(np.multiply(c / 2, np.sum(np.multiply(DP, HR), axis=1))) - np.sum(np.multiply(teta, DP))
# NP3 = np.sum(np.multiply(1 / c, OP)) + np.sum(np.multiply(c / 2, np.sum(np.multiply(DP, HP), axis=1)))
# NP4 = np.sum(np.multiply(1 / crm, ORM)) + np.sum(np.multiply(crm / 2, np.sum(np.multiply(drm, HRM), axis=1)))
# NP5 = np.sum(np.multiply(DP, TP)) + np.sum(np.multiply(DP, PCP)) + np.sum(np.multiply(DA, PCA)) + np.sum(
#     np.multiply(drm, PCR))
# NP6 = np.sum(np.multiply(fpp, FCP)) + np.sum(np.multiply(fpm, FCM)) + np.sum(np.multiply(fpa, FCA)) + np.sum(A)
# print(NP1)
# print(NP2)
# print(NP3)
# print(NP4)
# print(NP5)
# print(NP6)
# NP = NP1 - NP2 - NP3 - NP4 - NP5 - NP6
# print(NP)

class Individual(object):
    def __init__(self,var,obj):
        self.var = var
        self.obj_value = obj(self.var)
    def __str__(self):
        # return str(self.var) + "  " +str(self.obj_value)
        return str(self.obj_value)

def InitPopulation(N):
    population = []
    for i in range(N):
        A = np.random.uniform(0,10000,size=g)
        c = np.random.uniform(0,1,size=m)
        crm = np.random.uniform(0,1,size=s)
        drm = np.random.randint(0,2,size=s*l)
        fpp = np.random.randint(0,2,size=g)
        fpm = np.random.randint(0,2,size=j)
        fpa = np.random.randint(0,2,size=j*k)
        rho = np.random.uniform(0,1,size=g)
        x = []
        x.extend(A)
        x.extend(c)
        x.extend(crm)
        x.extend(drm)
        x.extend(fpp)
        x.extend(fpm)
        x.extend(fpa)
        x.extend(rho)

        x = np.array(x,dtype=float)         # shape = 35
        population.append(Individual(x,Objective))
    return population
def Crossover(x1,x2):
    a1 = Individual(np.concatenate([x1.var[:5],x2.var[5:]]),Objective)
    a2 = Individual(np.concatenate([x1.var[:10],x2.var[10:]]),Objective)
    a3 = Individual(np.concatenate([x1.var[:15],x2.var[15:]]),Objective)
    a4 = Individual(np.concatenate([x1.var[:20],x2.var[20:]]),Objective)
    a5 = Individual(np.concatenate([x1.var[:25],x2.var[25:]]),Objective)
    a6 = Individual(np.concatenate([x1.var[:30],x2.var[30:]]),Objective)
    return [a1,a2,a3,a4,a5,a6]

def Mutation(x):
    n = np.random.randint(low = 0,high=8)
    var_new = x.var
    if n==0:
        var_new[0:g] = np.random.uniform(0, 10000, size=g)
    elif n ==1:
        var_new[g:g+m] = np.random.uniform(0, 1, size=m)
    elif n == 2:
        var_new[g+m:g+m+s] = np.random.uniform(0, 1, size=s)
    elif n == 3:
        var_new[g+m+s:g+m+s+s*l] = np.random.randint(0, 2, size=s * l)
    elif n == 4:
        var_new[g+m+s+s*l:g+m+s+s*l+g] = np.random.randint(0, 2, size=g)
    elif n == 5:
        var_new[g+m+s+s*l+g:g+m+s+s*l+g+j] = np.random.randint(0, 2, size=j)
    elif n == 6:
        var_new[g+m+s+s*l+g+j:g+m+s+s*l+g+j+j*k] = np.random.randint(0, 2, size=j * k)
    elif n == 7:
        var_new[g+m+s+s*l+g+j+j*k:] = np.random.uniform(0, 1, size=g)
    return Individual(var_new,Objective)
def Selection(population,N):
    population.sort(key = lambda x: x.obj_value)
    return population[:N]


def Objective(x):       #g+m+s+s*l+g+j+j*k+g
    A = np.array(x[:g])
    c = np.array(x[g:g+m])
    crm = np.array(x[g+m:g+m+s])
    drm = np.reshape(np.array(x[g+m+s:g+m+s+s*l]),(s,l))
    fpp = np.array(x[g+m+s+s*l:g+m+s+s*l+g])
    fpm = np.array(x[g+m+s+s*l+g:g+m+s+s*l+g+j])
    fpa = np.reshape(np.array(x[g+m+s+s*l+g+j:g+m+s+s*l+g+j+j*k]),(j,k))
    rho = np.array(x[g+m+s+s*l+g+j+j*k:])
    # rho = np.array(x[-4:])
    # print(x.shape)
    # print(rho.shape)
    pw = pw0 - np.multiply(rho, DP)
    # print(pw)
    NP1 = np.sum(np.multiply(DP,pw))

    NP2 = np.sum(np.multiply(c/2,np.sum(np.multiply(DP,HR),axis=1))) - np.sum(np.multiply(teta,DP))
    NP3 = np.sum(np.multiply(1/c,OP))+np.sum(np.multiply(c/2,np.sum(np.multiply(DP,HP),axis=1)))
    NP4 = np.sum(np.multiply(1/crm,ORM))+np.sum(np.multiply(crm/2,np.sum(np.multiply(drm,HRM),axis=1)))
    NP5 = np.sum(np.multiply(DP,TP)) + np.sum(np.multiply(DP,PCP))+np.sum(np.multiply(DA,PCA))+np.sum(np.multiply(drm,PCR))
    NP6 = np.sum(np.multiply(fpp,FCP)) + np.sum(np.multiply(fpm,FCM))+np.sum(np.multiply(fpa,FCA))+np.sum(A)

    NP = NP1-NP2-NP3-NP4-NP5-NP6

    return -NP

if __name__ == "__main__":
    N = 1000
    mutation = 0.1
    crossover = 0.1
    iter = 1000
    population = InitPopulation(N)

    for i in range(1000):
        for _ in range(int(1000*mutation)):
            x1 = random.choice(population)
            x2 = random.choice(population)
            mu = Crossover(x1,x2)
            population.extend(mu)
        for _ in range(int(iter*crossover)):
            x = random.choice(population)
            x_new = Mutation(x)
            population.append(x_new)
            population.extend(mu)
        Selection(population,N)
        # print(Objective(population[0]))
        print(population[0])


# po = InitPopulation(10)
#
# x1 = po[0]
# x2 = po[1]
# print(Mutation(x1,x2))
#
#
# x1 = random.choice(po)
# x2 = random.choice(po)
#
# mutation = 0.1
#
# print(Mutation(x1,x2))
# for _ in range(100):
#     x1 = random.choice(po)
#     x2 = random.choice(po)
#     print(x1.var.shape)
#     print(Mutation(x1,x2))

# print(x1.var.shape)
# print(np.concatenate([x1.var[:10],x2.var[10:]]).shape)