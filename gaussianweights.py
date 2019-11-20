from scipy.spatial import cKDTree
import numpy as np

weightedflux = lambda flux, gw, nearest: np.sum(flux[nearest]*gw,axis=-1)

def gaussian_weights( X, w=None, neighbors=100, feature_scale=1000):
    '''
        K. Pearson: Gaussian weights of nearest neighbors
    '''
    if isinstance(w, type(None)): w = np.ones(X.shape[1])
    Xm = (X - np.median(X,0))/w
    kdtree  = cKDTree(Xm*feature_scale)
    nearest = np.zeros((X.shape[0],neighbors))
    gw  = np.zeros((X.shape[0],neighbors),dtype=np.float64)
    for point in range(X.shape[0]):
        ind = kdtree.query(kdtree.data[point],neighbors+1)[1][1:]
        dX  = Xm[ind] - Xm[point]
        Xstd= np.std(dX,0)       
        gX = np.exp(-dX**2/(2*Xstd**2))
        gwX  = np.product(gX,w)
        gw[point,:] = gwX/gwX.sum()
        nearest[point,:] = ind
    return gw, nearest.astype(int)
