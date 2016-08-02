import numpy as np
import matplotlib.pyplot as plt
from PyRenderer import Renderer
import sys, os
import os.path as osp
import scipy.io as spio
import trimesh
import mesh_util

exemplar_annotation_root = '/Users/amirrahimi/src/intrinsics/amir/new_annotations/car_annotations/'
cad_original_root = '/Users/amirrahimi/src/3Dshapes/pascal3d/'
cad_transformed_root = '/Users/amirrahimi/src/3Dshapes/pascal3d/transformed/' 

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
            px = viewpoint['px'][0,0][0,0]
            py = viewpoint['py'][0,0][0,0]
            yaw = viewpoint['theta'][0,0][0,0]
            distance_ratio = 2 #distance # TODO
            field_of_view = 45
            cad_idx = car['cad_index'][0,0]-1
            #x.initialize([cad_original_root + 'car%d.stl' % cad_idx ], viewport_size_x, viewport_size_y)
            
            print 'elevation %f' % elevation
            print 'azimuth %f' % azimuth
            print 'distance %f' % distance
            print 'yaw %f' % yaw
            print 'cad %i' % cad_idx
            
            mesh = trimesh.load_mesh(cad_transformed_root + 'car%d_t.stl' % cad_idx)
            ### mesh = trimesh.load_mesh(cad_original_root + 'car%d.stl' % cad_idx)

            e = elevation*np.pi/180
            a = azimuth*np.pi/180
            d = distance

            C = np.zeros((3,1))
            C[0] = 0 # d*np.cos(e)*np.sin(a)
            C[2] = -d # -d*np.cos(e)*np.cos(a)
            C[1] = 0 # -d*np.sin(e)

            roty = mesh_util.get_rot_y(-(azimuth-90)*np.pi/180)
            rotx = mesh_util.get_rot_x( elevation*np.pi/180)
            rotz = mesh_util.get_rot_z( yaw*np.pi/180)
            mesh.apply_transform( np.dot(rotx, roty) )
            
            mesh.apply_translation(C)
            mesh.apply_transform(rotz) #TODO: check -+
            
            mesh.export(file_obj='tmp.stl')
            mesh = trimesh.load_mesh('tmp.stl')
            # mesh.show()
            vs = mesh.vertices
            vs2d = vs.copy()
            f = 3000
            pad = 100
            vs2d[:,0] = -(vs[:,0]*f /(vs[:,2]) - px) + pad 
            vs2d[:,1] = vs[:,1]*f /(vs[:,2]) + py + pad 
            # vs2d[:,0] = viewport_size_x + pad - vs2d[:,0]
            a = np.zeros( (viewport_size_y+2*pad, viewport_size_x+2*pad) ) #+ distance
            a[np.int32(vs2d[:,1]), np.int32(vs2d[:,0]) ] =  np.sqrt( vs[:,0]**2 + vs[:,1]**2 + vs[:,2]**2 )
            plt.imshow(a)
            plt.show()

           
