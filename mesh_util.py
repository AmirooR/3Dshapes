import numpy as np
import trimesh
import cPickle as pickle
import GPy
import os

def write_ply(fname, faces, vertices):
    fd = open(fname, 'w')
    fd.write('ply\nformat ascii 1.0\n')
    fd.write('element vertex %d\n' % vertices.shape[0] )
    fd.write('property float x\n')
    fd.write('property float y\n')
    fd.write('property float z\n')
    fd.write('element face %d\n' % faces.shape[0] )
    fd.write('property list uchar int vertex_indices\n')
    fd.write('end_header\n')

    all3 = 3 * np.ones((faces.shape[0],1))
    np.savetxt(fd, vertices, fmt='%.5f')
    n_faces = faces - 1
    good_ply = np.hstack((all3, n_faces))
    np.savetxt(fd, good_ply.astype(np.uint32), fmt='%d')
    fd.close()

def get_rot_x(theta):
    return np.array( [[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]] )

def get_rot_y(theta):
    return np.array( [[np.cos(theta), 0, np.sin(theta), 0], [0, 1, 0, 0], [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]] )

def get_rot_z(theta):
    return np.array( [[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] )

def my_run_to_raw( shape, index_xy, index_z, **kwargs):
    raw = np.zeros( shape, dtype=np.bool)
    for xy, z in zip( index_xy, index_z):
        z_start = z[0]
        z_end = z[-1]
        raw[xy[0], xy[1]][z_start:z_end] = True
    return raw

def my_run_to_raw2D( shape, index_xy, index_z, **kwargs):
    raw = np.zeros( (shape[0], shape[1], 2)) #TODO uint16 is OK?
    for xy, z in zip( index_xy, index_z):
        z_start = z[0]
        z_end = z[-1]
        raw[xy[0], xy[1]][0] = z_start
        raw[xy[0], xy[1]][1] = z_end
    return raw

def mesh_to_voxelized_run2D( mesh, max_dim = 100, transformation = np.eye(4) ):
    final_X = np.zeros( (max_dim, max_dim, 2))
    mesh.apply_transform( transformation )
    mesh_lens = np.array( [mesh.bounds[1,i]-mesh.bounds[0,i] for i in range(3)] )
    max_len_idx = np.argmax( mesh_lens )
    scale = max_dim / mesh_lens[ max_len_idx ]
    scale -= 1e-6 # TODO: sometimes it adds +1 to the shape!which causes errors, so a small value is subtracted from scale
    m_origin = mesh.bounds.min(axis=0)
    mesh.apply_translation( - m_origin)
    mesh.apply_scale( scale )
    voxels = mesh.voxelized( 1 )
    filled_raw2D = my_run_to_raw2D( **voxels.run )
    final_X[0:filled_raw2D.shape[0], 0:filled_raw2D.shape[1], 0:filled_raw2D.shape[2]] = filled_raw2D
    return final_X

def transform_converted_matlab_meshes(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    roty = get_rot_y( -np.pi/2)
    rotz = get_rot_z( -np.pi/2)
    t = np.dot( rotz, roty)
    mesh.apply_transform(t)
    return mesh

def export_meshes_to_run2D_pkl(in_folder='/Users/amirrahimi/src/3Dshapes/car_train_colored/transformed/', skip_to=None):
    skip = True
    if skip_to is None:
        skip = False
    for x in os.listdir(in_folder):
        if x == skip_to:
            skip = False
        if x.endswith('stl') and not skip:
            print x
            mesh = trimesh.load_mesh(in_folder + x)
            run2D = mesh_to_voxelized_run2D(mesh, max_dim = 200)
            with open( in_folder + x[:-3] + 'pkl', 'wb') as f:
                pickle.dump(run2D, f, -1)

   
def export_matlab_meshes():
    for i in range(10):
        mesh_path = 'car%d.stl' % i
        mesh = transform_converted_matlab_meshes(mesh_path)
        mesh.export(file_obj='car%d_t.stl' % i)
        run2D = mesh_to_voxelized_run2D(mesh, max_dim = 200)
        with open('car%d_t.pkl' % i, 'wb') as f:
            pickle.dump(run2D, f, -1)
            
def GPLVM_experiment(ndim=200, input_dim=2, in_folder='/Users/amirrahimi/src/3Dshapes/car_train_colored/transformed/', num=105):
    Y = np.zeros((num, 200*200*2))
    i = 0
    for x in os.listdir(in_folder):
        if x.endswith('pkl'):
            with open(in_folder + x, 'rb') as f:
                a = pickle.load(f)
                Y[i,:] = a.ravel()
                i = i + 1
    y_mean = Y.mean(axis=0)
    y_var  = Y.var(axis=0)
    n0var_idx = y_var.nonzero()[0]
    Y_norm = (Y[:, n0var_idx] - y_mean[n0var_idx])/(y_var[n0var_idx])
    Q = input_dim
    m_gplvm = GPy.models.GPLVM( Y_norm, Q, kernel=GPy.kern.RBF(Q) )
    m_gplvm.optimize(messages=1, max_iters=5e4)
    return m_gplvm


def GPLVM_experiment_matlab(ndim=200, input_dim=2):
    Y = np.zeros((10, 200*200*2))
    for i in range(10):
        with open('car%d_t.pkl' % i, 'rb') as f:
            a = pickle.load(f)
            Y[i,:] = a.ravel()
    y_mean = Y.mean(axis=0)
    y_var  = Y.var(axis=0)
    n0var_idx = y_var.nonzero()[0]
    Y_norm = (Y[:, n0var_idx] - y_mean[n0var_idx])/(y_var[n0var_idx])
    Q = input_dim
    m_gplvm = GPy.models.GPLVM( Y_norm, Q, kernel=GPy.kern.RBF(Q) )
    m_gplvm.optimize(messages=1, max_iters=5e4)
    return m_gplvm

