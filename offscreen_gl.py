#!/usr/bin/env python
#-*- coding: UTF-8 -*-

import os, sys
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *

import math
import numpy

import numpy as np
import sys, os
import os.path as osp
import scipy.io as spio
from trimesh import load_mesh
from collections import namedtuple
import cv2

SceneStruct = namedtuple("SceneStruct", "meshes")
exemplar_annotation_root = '/Users/amirrahimi/src/intrinsics/amir/new_annotations/car_annotations/'
cad_original_root = '/Users/amirrahimi/src/3Dshapes/pascal3d/'
cad_transformed_root = '/Users/amirrahimi/src/3Dshapes/pascal3d/transformed/' 
images_root = '/Users/amirrahimi/src/intrinsics/amir/VOCdevkit/VOC2007/car_images/'

name = 'OpenGL viewer'
def get_rot_x(theta):
    return np.array( [[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]] )

def get_rot_y(theta):
    return np.array( [[np.cos(theta), 0, np.sin(theta), 0], [0, 1, 0, 0], [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]] )

def get_rot_z(theta):
    return np.array( [[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] )

def apply_transformations( mesh, azimuth, elevation, yaw, distance):
    d = distance
    C = np.zeros((3,1))
    C[0] = 0 # d*np.cos(e)*np.sin(a)
    C[2] = -d # -d*np.cos(e)*np.cos(a)
    C[1] = 0 # -d*np.sin(e)
    roty = get_rot_y(-(azimuth-90)*np.pi/180)
    rotx = get_rot_x( elevation*np.pi/180)
    rotz = get_rot_z( yaw*np.pi/180)
    mesh.apply_transform( np.dot(rotx, roty) )
    mesh.apply_translation(C)
    mesh.apply_transform(rotz)

class Offscreen():
    def __init__(self, maxWidth, maxHeight):
        self.meshes = []
        self.width = maxWidth
        self.height = maxHeight
        self.glutWin = -1
        glutInit(sys.argv)
        self.glutInitialized = True
        glutInitDisplayMode(GLUT_DEPTH, GLUT_SINGLE, GLUT_RGB) # TODO: RGB or RGBA?
        glutInitWindowSize(maxWidth, maxHeight)
        if self.glutWin < 0:
            glutWin = glutCreateWindow("OpenGL")
            glutHideWindow()
            self.fb = glGenFramebuffersEXT(1)
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fb)
            self.renderTex = glGenTextures(1)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.renderTex)
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, maxWidth, maxHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            
            self.depthTex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.depthTex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, maxWidth, maxHeight, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, None)

            self.fb2 = glGenFramebuffersEXT(1)
            glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, self.fb2)
            glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, self.renderTex, 0)
            glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, self.depthTex, 0)
            #glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT|GL_DEPTH_ATTACHMENT_EXT)
            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT)

        else:
            glutSetWindow(self.glutWin)

    def prepare_gl_buffers(self, mesh):
        """ Creates 3 buffer objets for each mesh, 
        to store the vertices, the normals, and the faces
        indices.
        """

        mesh.gl = {}
        vertices = np.asarray( mesh.vertices.copy(), dtype=np.float32)
        # Fill the buffer for vertex positions
        mesh.gl["vertices"] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, mesh.gl["vertices"])
        glBufferData(GL_ARRAY_BUFFER, 
                    vertices,
                    GL_DYNAMIC_DRAW)
        normals = np.asarray( mesh.vertex_normals.copy(), dtype=np.float32)

        # Fill the buffer for normals
        mesh.gl["normals"] = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, mesh.gl["normals"])
        glBufferData(GL_ARRAY_BUFFER, 
                    normals,
                    GL_DYNAMIC_DRAW)

        faces = np.asarray( mesh.faces.copy(), dtype=np.int32)
        # Fill the buffer for vertex positions
        mesh.gl["triangles"] = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.gl["triangles"])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                    faces,
                    GL_STATIC_DRAW)

        # Unbind buffers
        glBindBuffer(GL_ARRAY_BUFFER,0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0)

    
    def load_model(self, path, obj_idx=0):
        record = spio.loadmat(exemplar_annotation_root + path + '.mat', struct_as_record=True)
        objects = record['record']['objects'][0,0]
        classes = objects['class'][0]
        cur_obj_idx = -1
        mesh = None
        for i, x in enumerate(classes):
            if x == 'car':
                car = objects[0,i]
                if car['viewpoint'].shape == (1,1):
                    cur_obj_idx = cur_obj_idx + 1
                    if cur_obj_idx != obj_idx:
                        continue
                    viewpoint = car['viewpoint']
                    self.azimuth = viewpoint['azimuth'][0,0][0,0]
                    self.elevation = viewpoint['elevation'][0,0][0,0]
                    self.distance = viewpoint['distance'][0,0][0,0]
                    self.px = viewpoint['px'][0,0][0,0]
                    self.py = viewpoint['py'][0,0][0,0]
                    self.yaw = viewpoint['theta'][0,0][0,0]
                    self.focal = 3000
                    cad_idx = car['cad_index'][0,0]-1
                    print 'loading mesh car%d_t.stl' % cad_idx
                    far = 100
                    near = 0.001
                    px = self.px
                    py = self.py
                    focal = self.focal
                    self.near = near #TODO can do it by computing max/min of mesh Zs
                    self.far = far
                    self.perspMat = np.array([focal, 0, 0, 0, 0, -focal, 0, 0, -px, -py, near+far, -1, 0, 0, near*far, 0])
                    mesh = load_mesh(cad_transformed_root + 'car%d_t.stl' % cad_idx)
                    apply_transformations( mesh, self.azimuth, self.elevation, self.yaw, self.distance)

                    self.meshes.append(mesh)

        if mesh is not None:
            self.prepare_gl_buffers(mesh)
        else:
            print 'Error mesh is None!'

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glPushMatrix()

        for i, mesh in enumerate(self.meshes):
            #print 'recursive render of mesh %d' % i 
            material = {}
            self.apply_material(material)

            glBindBuffer(GL_ARRAY_BUFFER, mesh.gl["vertices"])
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, mesh.gl["normals"])
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0, None)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.gl["triangles"])
            glDrawElements(GL_TRIANGLES,len(mesh.faces) * 3, GL_UNSIGNED_INT, None)

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        glPopMatrix()

    def apply_material(self, mat):
        if not hasattr(mat, "gl_mat"): # evaluate once the mat properties, and cache the values in a glDisplayList.
            diffuse = numpy.array( [0.8, 0.8, 0.8, 1.0])
            specular = numpy.array([0., 0., 0., 1.0])
            ambient = numpy.array([0.2, 0.2, 0.2, 1.0])
            emissive = numpy.array([0., 0., 0., 1.0])
            shininess =  128
            wireframe = 0
            twosided = 1 #mat.properties.get("twosided", 1)

            mat["gl_mat"] = glGenLists(1)
            glNewList(mat["gl_mat"], GL_COMPILE)
    
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emissive)
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe else GL_FILL)
            glDisable(GL_CULL_FACE) if twosided else glEnable(GL_CULL_FACE)
    
            glEndList()
    
        glCallList(mat["gl_mat"])


    def camera_setup(self):
        glClearColor(0.1, 0.1, 0.1, 1.)
        ## glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHT0)
        #glutDisplayFunc(self.display)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, self.near, self.far)
        glMultMatrixd( self.perspMat)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0., 0., 0.,
                0., 0., -1.,
                0., 1., 0.)
        glPushMatrix()

    def drawPatchToDepthBuffer(self):
        glFlush()
        paddedWidth = self.width % 4
        if paddedWidth != 0:
            paddedWidth = 4 - paddedWidth + self.width
        else:
            paddedWidth = self.width

        dataBuffer_depth = (GLfloat * (paddedWidth * self.height)) (0.0)
        glReadPixels(0, 0, paddedWidth, self.height, GL_DEPTH_COMPONENT, GL_FLOAT, dataBuffer_depth)
        dataBuffer_rgb   = (GLubyte * (paddedWidth*self.height*3)) (0)
        glReadPixels(0, 0, paddedWidth, self.height, GL_RGB, GL_UNSIGNED_BYTE, dataBuffer_rgb)

        depths = np.asarray(dataBuffer_depth, dtype=np.float64)
        pixels = np.asarray(dataBuffer_rgb, dtype=np.uint8)
        patches = {'depths':depths, 'pixels':pixels}
        return patches

    def delete_mems(self):
        # clearing buffers
        glDeleteTextures(self.renderTex)
        glDeleteTextures(self.depthTex)
        glDeleteFramebuffersEXT(np.asarray([self.fb, self.fb2]))
        #glDeleteFramebuffersEXT(self.fb2)


if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print 'Usage: ' + __file__ + " <model>"
        sys.exit(0)
    img = cv2.imread(images_root + sys.argv[1] + '.jpg')
    width = img.shape[1]
    height = img.shape[0]
    
    for i in range(2):
        offscreen = Offscreen(width, height)
        offscreen.load_model(sys.argv[1], i)
        offscreen.camera_setup()
        offscreen.display()
        patches = offscreen.drawPatchToDepthBuffer()
        offscreen.delete_mems()
        with open('patches_depths_%d.Z' % i,'wb') as f:
            patches['depths'].tofile(f)
        with open('patches_pixels_%d.RGB' % i, 'wb') as f:
            patches['pixels'].tofile(f)
