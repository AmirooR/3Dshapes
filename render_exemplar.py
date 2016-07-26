import numpy as np
import matplotlib.pyplot as plt
from PyRenderer import Renderer
import sys, os
import os.path as osp
import scipy.io as spio

exemplar_annotation_root = '/Users/amirrahimi/src/intrinsics/amir/new_annotations/car_annotations/'
cad_original_root = '/Users/amirrahimi/src/3Dshapes/pascal3d/'

viewport_size_x = 500
viewport_size_y = 375

if len(sys.argv) != 2:
    print 'Usage: %s exemplar_name' % sys.argv[0]
    sys.exit()

name = sys.argv[1]
record = spio.loadmat(exemplar_annotation_root + name + '.mat', struct_as_record=True)
objects = record['record']['objects'][0,0]
classes = objects['class'][0]
for i, x in enumerate(classes):
    if x == 'car':
        car = objects[0,i]
        if car['viewpoint'].shape == (1,1):
            viewpoint = car['viewpoint']
            azimuth = viewpoint['azimuth'][0,0][0,0]
            elevation = viewpoint['elevation'][0,0][0,0]
            distance = viewpoint['distance'][0,0][0,0]
            yaw = 0
            distance_ratio = 2 #distance # TODO
            field_of_view = 45
            x = Renderer()
            cad_idx = car['cad_index'][0,0]-1
            x.initialize([cad_original_root + 'car%d.stl' % cad_idx ], viewport_size_x, viewport_size_y)
            print 'elevation %f' % elevation
            print 'azimuth %f' % azimuth
            print 'distance %f' % distance
            print 'cad %i' % cad_idx
            x.setViewpoint(-azimuth, elevation, yaw, distance_ratio, field_of_view)
            rendering, depth = x.render()
            rendering = rendering.transpose((2,1,0))
            #depth = depth.transpose((1,0))
            plt.imshow(rendering)
            plt.show()
