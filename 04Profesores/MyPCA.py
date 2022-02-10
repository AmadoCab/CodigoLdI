import numpy as np

def z_score(nd_array, axis=None):
    return (nd_array-np.mean(nd_array, axis=axis, keepdims=True))/np.std(nd_array, axis=axis, keepdims=True)

class PCA:
    # Attributes
    def __init__(self, n_components=False):
        self.n_components = n_components

    eigVal = None
    eigVec = None
    variance_ratio = None

    # Methods
    def __valid(self, data):
        if self.n_components:
            pass
        else:
            self.n_components = min(data.shape)

    def fit(self, data, rowvar=True):
        self.__valid(data)
        matrix = np.cov(data, rowvar=rowvar)
        eVal, eVec = np.linalg.eigh(matrix)
        eVal, eVec = eVal[::-1], eVec[:,::-1]
        self.eigVal, self.eigVec = eVal[0:self.n_components], eVec[:,0:self.n_components]
        self.variance_ratio = self.eigVal/np.sum(self.eigVal)

    def transform(self, data, axis=0, rowvar=True):
        if self.variance_ratio.any():
            if axis == 0:
                mat = np.array([])
                for Evec in np.transpose(self.eigVec):
                    row = np.array([])
                    for Fvec in np.transpose(data):
                        row = np.append(row,np.dot(Evec,Fvec))
                    mat = np.append(mat, -row)
                return mat.reshape((self.n_components,int(mat.shape[0]/self.n_components)))
            elif axis == 1:
                mat = np.array([])
                for Evec in np.transpose(self.eigVec):
                    row = np.array([])
                    for Fvec in data:
                        row = np.append(row,np.dot(Evec,Fvec))
                    mat = np.append(mat, -row)
                return mat.reshape((self.n_components,int(mat.shape[0]/self.n_components)))
            else:
                raise Exception("axis option must be 0 or 1")
        else:
            raise Exception("The model must be fitted first")

if __name__ == "__main__":
    print("Aquí vamos a hacer nuestra propia implementación de PCA")

    print("\nPrimero vamos a iniciar nuestros datos:")
    data = np.array([
            [2,4,6,6,7,5],
            [3,5,5,7,8,8]
        ])
    print(data)

    print("\nAhora vamos a estandarizar los datos:")
    data_st = z_score(data, axis=1)
    print(data_st)

    print("\nPor ultimo visualizamos los datos transformados")
    pca = PCA(n_components=1)
    pca.fit(data_st)
    print(pca.transform(data_st))

# EOF #

