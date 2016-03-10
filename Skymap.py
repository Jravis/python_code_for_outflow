#Plotting program for directional outflow that takes value form c file
import numpy as np
import healpy as hp
import numpy.linalg as la
import matplotlib.pyplot as plt
NSIDE =2 
import math as m
from numpy import ndarray
map1 = np.arange(hp.nside2npix(NSIDE))
"""
print hp.nside2npix(NSIDE),hp.nside2resol(NSIDE)
print hp.nside2pixarea(NSIDE)
print hp.get_map_size(m)
print hp.pixelfunc.get_min_valid_nside(60)
print (hp.pix2ang(NSIDE, 33))
"""
print map1
filename1 = "/home/mavrick/Documents/data_stacking/newinfall/data/cmass_result.153.122.dat"

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

pixNum = ndarray((48, 10), float)
partNum = ndarray((48, 10), float)
vel_net = ndarray((48, 10), float)
J = ndarray((10, 3), float)

id = np.genfromtxt(filename1, usecols=0, dtype=None)
for i in range(0, 10):
    filename = "/home/mavrick/Documents/data_stacking/newinfall/data/structure.5e11.%d.122.dat"% id[i]
#    print filename
    a = np.genfromtxt(filename, usecols=0, dtype=None)
    b = np.genfromtxt(filename, usecols=1, dtype=None)
    c = np.genfromtxt(filename, usecols=3, dtype=None)
    pixNum[:, i] = a
    partNum[:, i] = b
    vel_net[:, i] = c


#for i in range(0, 10):
J[:, 0] = np.genfromtxt(filename1, usecols=7, delimiter="\t", dtype=None)
J[:, 1] = np.genfromtxt(filename1, usecols=8, delimiter="\t", dtype=None)
J[:, 2] = np.genfromtxt(filename1, usecols=9, delimiter="\t", dtype=None)

clor = ["r", "b", "c", "m", "y", "k", "g", "v"]

temp = 0.0
fig = plt.figure(1)
for i in range(0, 6):
    theta = []
    for j in range(0, 48):
        pnum = pixNum[j, i]
#print pnum
        jx = J[i, 0]
        jy = J[i, 1]
        jz = J[i, 2]
        jVec = [jx, jy, jz]
        pixVec = hp.pix2vec(NSIDE, int(pnum))
        theta.append(m.degrees(py_ang(jVec, pixVec)))
    plt.plot(vel_net[:, i], theta, color=clor[i], marker="o", ms=5, linestyle="")
#plt.plot(partNum, theta,"bo")
plt.xlabel("Net Velocity")
plt.ylabel("theta")
plt.grid(True)
plt.savefig("/home/mavrick/Documents/data_stacking/newinfall/plots/Direction_outflow_multihalo.eps",  dpi=fig.dpi)
hp.mollview(map1, nest=False, title="Mollview image RING")
plt.show()
