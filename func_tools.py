"""
Standard Simplex Induced Clustering with
Hierarchical Deep Dictionary Learning

Application : FASHION MNIST
Authors     : Ulises Rodriguez Dominguez - CIMAT
              ulises.rodriguez@cimat.mx
----------------------------------------------------------
------           Auxiliary functions file            -----

"""
import munkres
import numpy as np
import gzip
from time import time
from munkres import Munkres

# Auxiliary functions (data reading and labels mapping) -------------------------------------
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# READ FASHION MNIST database images ------------------------------
# source:
# https://github.com/zalandoresearch/fashion-mnist/
# -----------------------------------------------------------------
def read_prepare_FASHION_MNIST(path):
    t0 = time()
    with gzip.open(path+'/train-labels-idx1-ubyte.gz','rb') as lbpath:
       y_tr = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(path+'/t10k-labels-idx1-ubyte.gz','rb') as lbpath:
       y_te = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(path+'/train-images-idx3-ubyte.gz','rb') as imgpath:
       X_tr = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(y_tr.shape[0],784)
    with gzip.open(path+'/t10k-images-idx3-ubyte.gz','rb') as imgpath:
       X_te = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(y_te.shape[0],784)
    t1 = time()
    print("Data read in %0.3fs" % (t1 - t0))
    y       = np.zeros((y_tr.shape[0]+y_te.shape[0],),dtype=np.int32)
    X       = np.zeros((y.shape[0],784),dtype=np.float32)
    y[0:y_tr.shape[0]] = y_tr[:]
    y[y_tr.shape[0]:y.shape[0]]  = y_te[:]
    X[0:y_tr.shape[0],:] = X_tr[:,:]
    X[y_tr.shape[0]:y.shape[0],:]  = X_te[:,:]
    print("Data prepare in %0.3fs" % (time() - t1))
    return X,y

# Code function by Binyuan Hui available at https://github.com/huybery/MvDSCN
# to map cluster labels L2 to given groundtruth labels L1
def mapLabels(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
