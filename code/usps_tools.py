import matplotlib
import matplotlib.pyplot as plt
import numpy as np


#fonction pour read USPS
def load_usps(filename):
    with open (filename , "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()

def normalize(data,data_min,data_max):
    data = (data-data_min)/(data_max-data_min)
    return data