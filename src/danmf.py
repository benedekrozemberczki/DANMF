import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy import sparse
from shallownmf import ShallowNMF
from sklearn.decomposition import NMF

class DANMF(object):
    """
    Deep autoencoder-like non-negative matrix factorization class.
    """
    def __init__(self, graph, args):
        """
        Initializing a DANMF object.
        :param graph: Networkx graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.A = nx.adjacency_matrix(self.graph)
        self.L = nx.laplacian_matrix(self.graph)
        self.D = self.L+self.A
        self.args = args
        self.p = len(self.args.layers)

    def setup_z(self, i):
        """
        Setup target matrix for pre-training process.
        """
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]

    def sklearn_pretrain(self,i):
        """
        Pretraining a single layer of the model with sklearn.
        :param i: Layer index.
        """
        nmf_model = NMF(n_components= self.args.layers[i], init="random", random_state=self.args.seed, max_iter = self.args.pre_iterations)
        U = nmf_model.fit_transform(self.Z)
        V = nmf_model.components_
        return U, V

    def general_pretrain(self,i):
        """
        Pretraining a single layer of the model with NMF custom class.
        :param i: Layer index.
        """
        nmf_model = ShallowNMF(X = self.Z, dimensions = self.args.layers[i], iterations = self.args.pre_iterations)
        nmf_model.update_weights()
        return nmf_model.U, nmf_model.V

    def pre_training(self):
        """
        Pre-training each NMF layer.
        """
        print("\nLayer pre-training started. \n")
        self.U_s = []
        self.V_s = []
        for i in tqdm(range(self.p), desc="Layers trained: ", leave=True):
            self.setup_z(i)
            if self.args.pre_training_method == "sklearn":
                U, V = self.sklearn_pretrain(i)
            elif self.args.pre_training_method == "shallow":
                U, V = self.general_pretrain(i)
            self.U_s.append(U)
            self.V_s.append(V)

    def setup_Q(self):
        """
        Setting up Q matrices.
        """
        self.Q_s = [None]*(self.p+1)
        self.Q_s[self.p]= np.eye(self.args.layers[self.p-1])
        for i in range(self.p-1,-1,-1):
            self.Q_s[i] = np.dot(self.U_s[i], self.Q_s[i+1])
   
    def update_U(self,i):
        """
        Updating left hand factors.
        :param i: Layer index.
        """
        if i == 0:
            R = self.U_s[0].dot(self.Q_s[1].dot(self.VpVpT).dot(self.Q_s[1].T))+self.A_sq.dot(self.U_s[0].dot(self.Q_s[1].dot(self.Q_s[1].T)))
            Ru = 2*self.A.dot(self.V_s[self.p-1].T.dot(self.Q_s[1].T))
            self.U_s[0] = (self.U_s[0]*Ru)/np.maximum(R,10**-10)
        else:
            R = self.P.T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.VpVpT).dot(self.Q_s[i+1].T)+self.A_sq.dot(self.P).T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.Q_s[i+1].T)
            Ru = 2*self.A.dot(self.P).T.dot(self.V_s[self.p-1].T).dot(self.Q_s[i+1].T)
            self.U_s[i] = (self.U_s[i]*Ru)/np.maximum(R,10**-10)

    def update_P(self,i):
        """
        Setting up P matrices.
        :param i: Layer index.
        """
        if i == 0:
           self.P = self.U_s[0]
        else:
           self.P = self.P.dot(self.U_s[i])

    def update_V(self,i):
        """
        Updating right hand factors.
        :param i: Layer index.
        """
        if (i < self.p-1):
            Vu = 2*self.A.dot(self.P).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])+self.V_s[i]
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd,10**-10)
        else:
            Vu = 2*self.A.dot(self.P).T+(self.args.lamb*self.A.dot(self.V_s[i].T)).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])+self.V_s[i]+(self.args.lamb*self.D.dot(self.V_s[i].T)).T
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd,10**-10)

    def calculate_cost(self, i):
        """
        Calculate loss.
        :param i: Global iteration.
        """
        reconstruction_loss_1 = np.linalg.norm(self.A-self.P.dot(self.V_s[-1]),ord= "fro")**2
        reconstruction_loss_2 = np.linalg.norm(self.V_s[-1]-self.A.dot(self.P).T,ord= "fro")**2
        regularization_loss = np.trace(self.V_s[-1].dot(self.L.dot(self.V_s[-1].T)))
        self.loss.append([i+1, reconstruction_loss_1,reconstruction_loss_2,regularization_loss])

    def save_embedding(self):
        """
        Save embedding matrix.
        """
        embedding = np.concatenate([np.array(range(self.P.shape[0])).reshape(-1,1), self.P, self.V_s[-1].T], axis = 1)
        columns = ["id"] + ["x_" + str(x) for x in range(self.args.layers[-1]*2)]
        embedding = pd.DataFrame(embedding, columns = columns)
        embedding.to_csv(self.args.output_path, index = None)

    def save_membership(self):
        """
        Save cluster membership.
        """
        index = np.argmax(self.P,axis=1)
        self.membership = {int(i):int(index[i]) for i in range(len(index))}
        with open(self.args.membership_path,"w") as f:
            json.dump(self.membership, f)
      
    def training(self):
        """
        Training process after pre-training.
        """
        print("\n\nTraining started. \n")
        self.loss = []
        self.A_sq = self.A.dot(self.A.T)
        for iteration in tqdm(range(self.args.iterations), desc="Training pass: ", leave = True):
            self.setup_Q()
            self.VpVpT = self.V_s[self.p-1].dot(self.V_s[self.p-1].T)
            for i in range(self.p):
                self.update_U(i)
                self.update_P(i)
                self.update_V(i)
            if self.args.calculate_loss:
                self.calculate_cost(iteration)
        self.save_membership()
        self.save_embedding()
