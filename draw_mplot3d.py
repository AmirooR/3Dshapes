import os, sys
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) != 2:
    print "Usage: %s run2D_model.pkl" % sys.argv[0]
    sys.exit()

with open(sys.argv[1], 'rb') as f:
    model = pickle.load(f)
    r,c = np.nonzero( model[:,:,1])
    xs = np.hstack((r,r))
    ys = np.hstack((c,c)) # np.zeros((num_points,))
    zs = np.zeros(len(ys))
    zs[:len(r)] = model[r,c,0]
    zs[-len(r):] = model[r,c,1]
    plt.imshow(model[:,:,1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, depthshade=True)
    plt.show()
