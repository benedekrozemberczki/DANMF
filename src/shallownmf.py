import numpy as np
import networkx as nx
from scipy import sparse
from tqdm import tqdm

class ShallowNMF(object):
    
    def __init__(self, X, dimensions, iterations):
        self.X = X
        self.dimensions = dimensions
        self.iterations = iterations
        self.rows = self.X.shape[0]
        self.columns =  self.X.shape[1]
        self.init_weights()

    def init_weights(self):
        self.U = np.random.uniform(0, 1, (self.rows, self.dimensions))
        self.V = np.random.uniform(0, 1, (self.dimensions, self.columns))

    def update_U(self):
        enum = 2*self.X.dot(self.V.T)*self.U
        denom = self.U.dot(self.V.dot(self.V.T))+self.X.dot(self.X.T.dot(self.U))
        denom[denom< 10**-15] = 10**-15
        self.U = enum/denom
 
    def update_V(self):
        enum = (self.X.T.dot(2*self.U)*self.V.T).T
        denom = self.U.T.dot(self.U.dot(self.V))+self.V
        denom[denom< 10**-15] = 10**-15
        self.V = enum/denom

    def update_weights(self):
        for i in tqdm(range(self.iterations),desc="Layer training round: "):
            self.update_U()
            self.update_V()
