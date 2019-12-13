import numpy as np
from vmi_parameter import *
import random

DA = np.ones((j,k))                     # j*k
DP = np.ones((m,g))            # m*g
pw = pw0
W = 0.5
c1 = 0.8
c2 = 0.9

class Individual(object):
    def __init__(self,var,obj):
        self.var = var
        self.pvar = self.var
        # self.gbest = self.var
        self.obj_value = obj(self.var)
        # self.obj_gvalue = 0
        self.obj_value_p = obj(self.pvar)
        self.velocity = 0
    def __str__(self):
        # return str(self.var) + "  " +str(self.obj_value)
        return str(self.obj_value)
    def move(self):
        self.var[: g + m + s + g] = self.var[: g + m + s + g] + self.velocity
def InitPopulation(N):
    population = []
    for i in range(N):
        A = np.random.uniform(0,10000,size=g)
        c = np.random.uniform(0,1,size=m)
        crm = np.random.uniform(0,1,size=s)
        rho = np.random.uniform(0, 1, size=g)

        drm = np.random.randint(0,2,size=s*l)
        fpp = np.random.randint(0,2,size=g)
        fpm = np.random.randint(0,2,size=j)
        fpa = np.random.randint(0,2,size=j*k)

        x = []
        x.extend(A)
        x.extend(c)
        x.extend(crm)
        x.extend(rho)

        x.extend(drm)
        x.extend(fpp)
        x.extend(fpm)
        x.extend(fpa)


        x = np.array(x,dtype=float)         # shape = 35
        population.append(Individual(x,Objective))
    return population



def Objective(x):       #g+m+s+s*l+g+j+j*k+g
    A = np.array(x[:g])
    c = np.array(x[g:g+m])
    crm = np.array(x[g+m:g+m+s])

    rho = np.array(x[g + m + s : g + m + s + g])


    drm = np.reshape(np.array(x[g+m+s+ g:g+m+s+s*l+ g]),(s,l))
    fpp = np.array(x[g+m+s+s*l+ g:g+m+s+s*l+g+ g])
    fpm = np.array(x[g+m+s+s*l+g:g+m+s+s*l+g+j])
    fpa = np.reshape(np.array(x[g+m+s+s*l+g+j+ g:g+m+s+s*l+g+j+j*k+ g]),(j,k))

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

    return NP


class Space():

    def __init__(self,population, N,obj):
        self.population = population
        self.N = N
        self.obj = obj
        self.gvar = 0
        self.obj_value_g = 0
    def print_particles(self):
        for p in self.population:
            print(p)

    def fitness(self, x):
        return self.obj(x.var)

    def set_pbest(self):
        for p in self.population:
            # fitness_cadidate = self.fitness(p)
            if (p.obj_value_p < p.obj_value):
                p.obj_value_p = p.obj_value
                p.pvar = p.var

    def set_gbest(self):
        for p in self.population:
            if (self.obj_value_g > p.obj_value):
                self.obj_value_g = p.obj_value
                self.gvar = p.var

    def move_particles(self):
        for particle in self.population:
            global W
            new_velocity = (W * particle.velocity) + (c1 * random.random()) * (
                        particle.pvar[:g+m+s+g] - particle.var[:g+m+s+g]) + \
                           (random.random() * c2) * (self.gvar[:g+m+s+g] - particle.var[:g+m+s+g])
            particle.velocity = new_velocity
            particle.move()




if __name__ == "__main__":
    N = 1000

    iter = 1000
    population = InitPopulation(N)

    search_space = Space(population,N,Objective)

    # search_space.print_particles()

    iteration = 0
    while (iteration < iter):
        search_space.set_pbest()
        search_space.set_gbest()

        search_space.move_particles()
        iteration += 1
    print(Objective(search_space.gvar))
    print("The best solution is: ", search_space.gvar, " in n_iterations: ", iteration)

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