#!/usr/bin/env python
#-*- coding: UTF-8 -*-

import os, sys
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *

import math
import numpy

import numpy as np
import sys, os
import os.path as osp
import scipy.io as spio
from trimesh import load_mesh
from collections import namedtuple
#import skimage.io as skio
import cv2
import time

SceneStruct = namedtuple("SceneStruct", "meshes")
exemplar_annotation_root = '/Users/amirrahimi/src/intrinsics/amir/new_annotations/car_annotations/'
cad_original_root = '/Users/amirrahimi/src/3Dshapes/pascal3d/'
cad_transformed_root = '/Users/amirrahimi/src/3Dshapes/pascal3d/transformed/' 
images_root = '/Users/amirrahimi/src/intrinsics/amir/VOCdevkit/VOC2007/car_images/'

name = 'OpenGL viewer'

### NOTE: read them by image!
#height = 333
#width = 500
#viewport_size_x = width
#viewport_size_y = height


def get_rot_x(theta):
    return np.array( [[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]] )

def get_rot_y(theta):
    return np.array( [[np.cos(theta), 0, np.sin(theta), 0], [0, 1, 0, 0], [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]] )

def get_rot_z(theta):
    return np.array( [[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] )


class GLRenderer():
    def __init__(self):

        self.scene = None

        self.using_fixed_cam = False
        self.current_cam_index = 0
        self.use_texture = False
        # for FPS calculation
        self.prev_time = 0
        self.prev_fps_time = 0
        self.frames = 0

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

    def orthogonalStart(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(-width/2, width/2, -height/2, height/2)
        glMatrixMode(GL_MODELVIEW)

    def orthogonalEnd(self):
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def background(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)
        self.orthogonalStart()
        iw = width
        ih = height
        glPushMatrix()
        glTranslatef( -iw/2, -ih/2, 0 )
        glBegin(GL_QUADS)
        glTexCoord2i(0,0)
        glVertex2i(0, 0)
        glTexCoord2i(1,0) 
        glVertex2i(iw, 0)
        glTexCoord2i(1,1) 
        glVertex2i(iw, ih)
        glTexCoord2i(0,1) 
        glVertex2i(0, ih)
        glEnd()
        glPopMatrix()
        self.orthogonalEnd();

    def load_texture(self, image_path):
        #img = skio.imread(image_path)
        img = cv2.imread(images_root + image_path + '.jpg')
        img_r = np.flipud(cv2.resize(img, (width, height)))
        texture = glGenTextures(1)
        glBindTexture( GL_TEXTURE_2D, texture ); 
        glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE ); 
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_NEAREST);

        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,GL_BGR, GL_UNSIGNED_BYTE, img_r.ravel()) # or img itself ?
        return texture


    def load_model(self, path, postprocess = None):
        self.scene = SceneStruct( meshes = [] )
        record = spio.loadmat(exemplar_annotation_root + path + '.mat', struct_as_record=True)
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
                    focal = 3000
                    cad_idx = car['cad_index'][0,0]-1
                    print 'loading mesh car%d_t.stl' % cad_idx
                    far = 100
                    near = 0.001
                    self.near = near
                    self.far = far
                    r = viewport_size_x - px
                    l = -px
                    t = viewport_size_y - py
                    b = -py
                    self.perspMat = np.array([focal, 0, 0, 0, 0, -focal, 0, 0, -px, -py, near+far, -1, 0, 0, near*far, 0])
                    ###self.perspMat = np.array([focal, 0, 0, 0, 0, -focal, 0, 0, -px, -height/2, near+far, -1, 0, 0, near*far, 0])
                    mesh = load_mesh(cad_transformed_root + 'car%d_t.stl' % cad_idx)
                    print 'Done'
                    e = elevation*np.pi/180
                    a = azimuth*np.pi/180
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
                    self.scene.meshes.append(mesh)
                    self.bb_min = mesh.bounds[0,:]
                    print self.bb_min
                    self.bb_max = mesh.bounds[1,:]
                    print self.bb_max

                    self.scene_center = [(a + b) / 2. for a, b in zip(self.bb_min, self.bb_max)]
                    print self.scene_center


        for index, mesh in enumerate(self.scene.meshes):
            print 'preparing buffers'
            self.prepare_gl_buffers(mesh)
            print '%d done' % index

    def set_default_camera(self):
        if not self.using_fixed_cam:
            glLoadIdentity()
            gluLookAt(0.,0.,0.,
                      0.,0.,-1.,
                      0., 1.,0.)

    def set_camera(self, camera):

        if not camera:
            return

        self.using_fixed_cam = True

        znear = camera.clipplanenear
        zfar = camera.clipplanefar
        aspect = camera.aspect
        fov = camera.horizontalfov

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Compute gl frustrum
        tangent = math.tan(fov/2.)
        h = znear * tangent
        w = h * aspect

        # params: left, right, bottom, top, near, far
        glFrustum(-w, w, -h, h, znear, zfar)
        # equivalent to:
        #gluPerspective(fov * 180/math.pi, aspect, znear, zfar)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cam = transform(camera.position, camera.transformation)
        at = transform(camera.lookat, camera.transformation)
        gluLookAt(cam[0], cam[2], -cam[1],
                   at[0],  at[2],  -at[1],
                       0,      1,       0)

    def fit_scene(self, restore = False):
        """ Compute a scale factor and a translation to fit and center 
        the whole geometry on the screen.
        """
        print 'fit_scene called!'
        x_max = self.bb_max[0] - self.bb_min[0]
        y_max = self.bb_max[1] - self.bb_min[1]
        tmp = max(x_max, y_max)
        z_max = self.bb_max[2] - self.bb_min[2]
        tmp = max(z_max, tmp)

        if not restore:
            tmp = 1. / tmp

        #logger.info("Scaling the scene by %.03f" % tmp)
        glScalef(tmp, tmp, tmp)
        print 'scale %f' % tmp

        # center the model
        direction = -1 if not restore else 1
        print 'direction %d' % direction
        print 'center %f, %f, %f' % ( self.scene_center[0], self.scene_center[1], self.scene_center[2])
        print 'x_max, y_max, z_max = (%f, %f, %f)' % (x_max, y_max, z_max)
        glTranslatef( direction * self.scene_center[0], 
                      direction * self.scene_center[1], 
                      direction * self.scene_center[2] )

        return x_max, y_max, z_max

    def apply_material(self, mat):
        """ Apply an OpenGL, using one OpenGL display list per material to cache 
        the operation.
        """

        if not hasattr(mat, "gl_mat"): # evaluate once the mat properties, and cache the values in a glDisplayList.
            diffuse = numpy.array( [0.8, 0.8, 0.8, .5])
            specular = numpy.array([0., 0., 0., 0.5])
            ambient = numpy.array([0.2, 0.2, 0.2, 0.5])
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

    
   
    def do_motion(self):
        #gl_time = glutGet(GLUT_ELAPSED_TIME)
        #self.angle = (gl_time - self.prev_time) * 0.1
        #self.prev_time = gl_time
        # Compute FPS
        #self.frames += 1
        
        glutPostRedisplay()

    def recursive_render(self, node):
        """ Main recursive rendering method.
        """

        # save model matrix and apply node transformation
        glPushMatrix()
        #m = node.transformation.transpose() # OpenGL row major
        #glMultMatrixf(m)

        for i, mesh in enumerate(node.meshes):
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

        #for child in node.children:
        #    self.recursive_render(child)

        glPopMatrix()


    def display(self):
        """ GLUT callback to redraw OpenGL surface
        """
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        if self.use_texture:
            glDisable(GL_DEPTH_TEST)
            glLoadIdentity()
            glEnable(GL_TEXTURE_2D)
            self.background()
            glDisable(GL_TEXTURE_2D)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)

        #glRotatef(self.angle,0.,1.,0.)
        self.recursive_render(self.scene)

        glutSwapBuffers()
        self.do_motion()
        return

    ####################################################################
    ##               GLUT keyboard and mouse callbacks                ##
    ####################################################################
    def onkeypress(self, key, x, y):
        if key == 'q':
            sys.exit(0)
        elif key == 's':
            print 'reading depths'
            buffer_ = (GLfloat * (viewport_size_y*viewport_size_x) )(0.0)
            glReadPixels(0, 0, viewport_size_x, viewport_size_y, GL_DEPTH_COMPONENT, GL_FLOAT, buffer_)
            print 'buffer size: %d' % len(buffer_)
            print 'buffer[0] %f' % buffer_[0]
            depths = np.asarray(buffer_, dtype=np.float64)
            with open('depths.Z','wb') as f:
                depths.tofile(f)
        elif key == 'r':
            print 'going right'
            mesh = self.scene.meshes[-1]
            vertices = np.asarray( mesh.vertices.copy(), dtype=np.float32)
            vertices[:,0] = vertices[:,0] + 0.01
            mesh.vertices[:,0] += 0.01
            glBindBuffer(GL_ARRAY_BUFFER, mesh.gl["vertices"])
            #glBufferData(GL_ARRAY_BUFFER, 
            #        vertices,
            #        GL_DYNAMIC_DRAW)
            glBufferSubData(target=GL_ARRAY_BUFFER, offset=0, size=None, data=vertices)
            #glBindBuffer(GL_ARRAY_BUFFER,0)
        elif key == 'm':
            modelviewmatrix=glGetDoublev(GL_MODELVIEW_MATRIX)

            print 'ModelViewMatrix'
            print modelviewmatrix

            projectionmatrix=glGetDoublev(GL_PROJECTION_MATRIX)
            print 'ProjectionMatrix'
            print projectionmatrix

            viewport=glGetIntegerv(GL_VIEWPORT)

            print 'Viewport'
            print viewport
            x = 400
            y = 100
            print float(x), viewport[3]-float(y)

            unproject_mat = np.linalg.inv( projectionmatrix.T) # NOTE: modelview is identity here, opengl is row order
            new_p_ndc = np.array([ 2*float(x)/viewport_size_x - 1,  1 - 2*float(y)/viewport_size_y, 1., 1.]) # 2*z -1
            new_p_world = np.dot( unproject_mat, new_p_ndc)
            print 'new_p_world'
            print new_p_world/new_p_world[3]

            wx,wy,wz= gluUnProject(float(x),viewport[3]-float(y),1.,
                                              modelviewmatrix,
                                              projectionmatrix,
                                              viewport)

            print 'World Coordinate'
            print wx,wy,wz
            

    def render(self, filename=None, fullscreen = False, autofit = False, postprocess = None, use_texture = False):
        """

        :param autofit: if true, scale the scene to fit the whole geometry
        in the viewport.
        """
    
        # First initialize the openGL context
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        if not fullscreen:
            glutInitWindowSize(width, height)
            glutCreateWindow(name)
        else:
            glutGameModeString("1024x768")
            if glutGameModeGet(GLUT_GAME_MODE_POSSIBLE):
                glutEnterGameMode()
            else:
                print("Fullscreen mode not available!")
                sys.exit(1)

        self.use_texture = use_texture

        self.load_model(filename, postprocess = postprocess)
        if self.use_texture:
            self.texture = self.load_texture(filename) 


        glClearColor(0.1,0.1,0.1,1.)
        #glShadeModel(GL_SMOOTH)

        glEnable(GL_LIGHTING)

        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)

        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHT0)

        glutDisplayFunc(self.display)


        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        #gluPerspective(35.0, width/float(height) , 0.10, 100.0)
        glOrtho(0, viewport_size_x, viewport_size_y, 0, self.near, self.far)
        glMultMatrixd( self.perspMat)
        glMatrixMode(GL_MODELVIEW)
        self.set_default_camera()

        #if autofit:
            # scale the whole asset to fit into our view frustum·
        #    self.fit_scene()

        glPushMatrix()

        glutKeyboardFunc(self.onkeypress)
        glutIgnoreKeyRepeat(1)

        glutMainLoop()


if __name__ == '__main__':
    if not len(sys.argv) > 1:
        print("Usage: " + __file__ + " <model>")
        sys.exit(0)
    img = cv2.imread(images_root + sys.argv[1] + '.jpg')
    width = img.shape[1]
    height = img.shape[0]
    viewport_size_x = width
    viewport_size_y = height
    glrender = GLRenderer()
    glrender.render(sys.argv[1], fullscreen = False, postprocess = None, use_texture = True)

