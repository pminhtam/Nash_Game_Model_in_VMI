import cvxpy as cp
import numpy as np


class retail:
    def __init__(self):
        ## 1
        n = 3
        g = 4
        self.eA = eA    # m*g*g
        self.ea = ea    # m*g*m*g
        self.ep = ep    # m*g*m*g
        self.K = K      # m*g
        self.v = v      # m*g*m*g
        self.u = u      # m*g*g
        self.beta = beta    # m*g*m*g
        ## 2
        self.pw0 = pw0  # g
        ## 3
        self.w = w      # m
        ## 4

        ## 5
        self.p = cp.Variable()      # m*g
        self.pw = pw    # m*g
        self.teta = teta    #
        self.a = cp.Variable()
    def make_prob(self, m_retail):
        self.DP = self.K[m_retail] + cp.sum(self.u[m_retail]*cp.power(self.A,self.eA))+ sum(self.beta*cp.power(self.p,self.ep) + self.v*cp.power(self.a,self.ea))
        self.pw = self.pw0
        self.NP = cp.sum(self.DP) - cp.sum(self.DP*self.pw) - cp.sum(self.teta*self.DP) - cp.sum(self.a)

        self.constraints = []




