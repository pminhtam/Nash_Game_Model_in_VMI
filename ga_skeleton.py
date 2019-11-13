import numpy as np
import random

class Individual(object):
    def __init__(self,var,obj):
        self.var = var
        self.obj_value = obj(self.var)
    def __str__(self):
        return str(self.var) + "  " +str(self.obj_value)

def InitPopulation(N):
    population = []
    for i in range(N):
        population.append(Individual(np.random.uniform(-10,10),Objective))
    return population
def Mutation(x1,x2):
    a1 = Individual((x1.var+x2.var)/2,Objective)
    a2 = Individual(x1.var-x2.var,Objective)
    a3 = Individual(x2.var-x1.var,Objective)
    a4 = Individual(x1.var+x2.var,Objective)
    return [a1,a2,a3,a4]

def Selection(population,N):
    population.sort(key = lambda x: x.obj_value)
    return population[:N]

def Objective(var):
    return np.abs(var*5+var**4-15*var**3+10)


if __name__ == "__main__":
    N = 10000
    mutation = 0.01

    iter = 1000
    population = InitPopulation(N)

    for i in range(1000):
        for j in range(int(10000*mutation)):
            x1 = random.choice(population)
            x2 = random.choice(population)
            mu = Mutation(x1,x2)
            population.extend(mu)
        Selection(population,N)
        print(population[0])