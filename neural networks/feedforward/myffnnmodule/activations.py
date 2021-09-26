import numpy as np

class ReLu:
    def activate(Z):
        return np.maximum(Z,0)
    def derivative(Z):
        return Z > 0

class sigmoid:
    def activate(Z):
        return 1/(1+np.exp(-Z))
    def derivative(Z):
        sig = 1/(1+np.exp(-Z))
        return sig*(1-sig)

class LeakyReLu:
    def activate(Z):
        return np.maximum(0.01*Z,Z)
    def derivative(Z):
        dZ = np.zeros(shape=Z.shape)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z[i][j] < 0:
                    dZ[i][j] = 0.01
                else:
                    dZ[i][j] = 1
        return dZ

class softMax:
    def activate(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A
