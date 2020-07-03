import os
import pdb
import math
import numpy as np 
import torch
import scipy.io as sio
from torch.utils import data
from metric import nmse_metric_np as nmse_metric


def loadDataSet(nums, size, prob, snr, test_nums=50):
    '''
    this function calculate the real part and the imag part of the complex data respectively
    '''
    dataPath = "../matlabAPI/datas_n{:.0f}_s{:.0f}_p{:.0f}_snr{:.0f}.mat".format(nums,size,prob*100,snr)
    labelPath = "../matlabAPI/labels_n{:.0f}_s{:.0f}_p{:.0f}_snr{:.0f}.mat".format(nums,size,prob*100,snr)
    thetaPath = "../matlabAPI/theta.mat"
    phiPath = "../matlabAPI/phi.mat"
    #dataPath = '../matlabAPI/2e5_datas_s256_snr10000.mat'
    #labelPath = '../matlabAPI/2e5_labels_s256_snr10000.mat'
    load_data = sio.loadmat(dataPath)
    load_label = sio.loadmat(labelPath)
    load_theta = sio.loadmat(thetaPath)
    load_phi = sio.loadmat(phiPath)

    comDatas = load_data['datas']
    comLabels = load_label['labels']
    theta = load_theta['theta']
    phi = load_phi['phi']

    idx = np.arange(len(comDatas))
    #np.random.shuffle(idx)
    selected_idx = idx[:test_nums]
    datas = comDatas[selected_idx]
    labels = 1 - comLabels[selected_idx]

    return datas, labels, theta[0], phi[0]



def calMse(b, x_hat, A, size_average=True):
    b_hat = np.dot(A, x_hat)
    mse = np.linalg.norm(b - b_hat)**2
    if size_average:
        mse = mse / len(b)
    return mse

def calGrad(b, x_hat, A):
    b_hat = np.dot(A, x_hat)
    error = (b_hat - b)
    grad = np.dot(np.matrix.getH(A), error)
    return grad

def runLeastSquare():
    antenna_x = 16
    antenna_y = 16
    sample_nums = 2e5
    antenna_size = antenna_x * antenna_y
    fault_prob = 0.1
    SNR = 10000
    sample_size = antenna_size
    test_nums = 50
    maxEpoch = int(5e7)  
    lr = 6e-4
    
    datas, labels, theta, phi = loadDataSet(sample_nums, antenna_size, fault_prob, SNR, test_nums=test_nums)
    for measurements in range(sample_size, 20, -10):
        A = np.zeros((measurements, antenna_size)) + 1j*np.zeros((measurements, antenna_size))
        m = np.arange(antenna_x)
        n = np.arange(antenna_y)
        for i in range(measurements):
            ax = np.exp(1j * m * np.pi * np.sin(theta[i]) * np.cos(phi[i]))
            ay = np.exp(1j * n * np.pi * np.sin(theta[i]) * np.sin(phi[i]))
            A[i,:] = np.kron(ax,ay)
        for test_id in range(test_nums):
            b = datas[test_id,:measurements]
            b = b[:,np.newaxis]
            x = labels[test_id]
            x = x[:,np.newaxis]
            
            # initial x_hat
            x_hat = np.random.random(x.shape) + 1j* np.random.random(x.shape)
            for epoch in range(maxEpoch):
                nmse = nmse_metric(x_hat, x)
                mse = calMse(b, x_hat, A,)
                if epoch % 20000 == 0:
                    print("Epoch: {epoch}/{maxEpoch} | Nmse:{nmse:6f} | Mse:{mse:.6f}".format(epoch=epoch+1, maxEpoch=maxEpoch, nmse=nmse, mse=mse))
                grad = calGrad(b, x_hat, A)
                #pdb.set_trace()
                x_hat = x_hat - lr * grad
            pdb.set_trace()





if __name__ == '__main__':
    runLeastSquare()