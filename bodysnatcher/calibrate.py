import numpy as np
import cv2
import itertools
import math
import sys
import base64

from .util import flipDepthFrame

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import CudaKdePacketPipeline

FRAME_WIDTH = 512
FRAME_HEIGHT = 424

CHESS_HEIGHT = 6
CHESS_WIDTH = 13
ALL_CHESS = CHESS_HEIGHT*CHESS_WIDTH

#using aaxa P5 projector throw ratio
F_X = 1.47*1280.
F_Y = F_X
C_X = 640
C_Y = 360
IMAGE_WIDTH = 1280.
IMAGE_HEIGHT = 720.
FAR_PLANE= 1000.
NEAR_PLANE= 0.01

camera_matrix = np.array([[F_X, 0, C_X],
                          [0, F_Y, C_Y],
                          [0, 0, 1]], dtype = "double")

def openGLProjMat():
    out = [[2.0* F_X / IMAGE_WIDTH, 0, 2.0 * (C_X/IMAGE_WIDTH) - 1.0, 0],
           [0, 2.0 * F_Y / IMAGE_HEIGHT, 2.0*(C_Y/IMAGE_HEIGHT) - 1.0, 0],
           [0, 0, -(FAR_PLANE + NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE),
            -2.0 * FAR_PLANE * NEAR_PLANE / (FAR_PLANE - NEAR_PLANE)],
           [0, 0, -1, 0]]
    out = np.array(out, dtype = np.float);
    return np.transpose(out).reshape(16).tolist()

def openGLViewMat(rot, trans):
    rotMat, _ = cv2.Rodrigues(rot)
    flipAxis = np.array([[1., 0., 0.],[0., -1., 0.], [0., 0., -1.]])
    rotMat = np.matmul(flipAxis, rotMat)
    trans = np.matmul(flipAxis, trans)
    out = np.hstack((rotMat, trans))
    out = np.vstack((out, np.array([[0, 0, 0, 1.0]])))
    return np.transpose(out).reshape(16).tolist()

def map3Dto2D(point, rot, trans):
    lstPoints = point.tolist()
    if len(point) == 1:
        print point.shape
        lstPoints = point[0].tolist()
    x, y, z = lstPoints
    rotMat, _ = cv2.Rodrigues(rot)
    t = np.matmul(np.hstack((rotMat, trans)), np.array([[x],[y],[z],[1]]))
    res = np.matmul(camera_matrix, t)
    u, v, l = res.tolist()
    u = u[0] / l[0]
    v = v[0] / l[0]
    print 'u {0}'.format(u)
    print 'v {0}'.format(v)
    return (u,v)

def initCamera(fn, pipeline):
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)
    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)
    types = (FrameType.Color | FrameType.Ir | FrameType.Depth)
    listener = SyncMultiFrameListener(types)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.startStreams(rgb= True, depth=True)
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())
    return (device, listener, registration)


def warmUp(listener):
    count = 0
    while count < 50:
        frames = listener.waitForNewFrame()
        count = count + 1
        listener.release(frames)

def shutdownCamera(device):
    device.stop()
    device.close()


image_points = np.array([
    (160, 160), (160, 240), (160, 320), (160, 400), (160, 480), (160, 560),
    (240, 160), (240, 240), (240, 320), (240, 400), (240, 480), (240, 560),
    (320, 160), (320, 240), (320, 320), (320, 400), (320, 480), (320, 560),
    (400, 160), (400, 240), (400, 320), (400, 400), (400, 480), (400, 560),
    (480, 160), (480, 240), (480, 320), (480, 400), (480, 480), (480, 560),
    (560, 160), (560, 240), (560, 320), (560, 400), (560, 480), (560, 560),
    (640, 160), (640, 240), (640, 320), (640, 400), (640, 480), (640, 560),
    (720, 160), (720, 240), (720, 320), (720, 400), (720, 480), (720, 560),
    (800, 160), (800, 240), (800, 320), (800, 400), (800, 480), (800, 560),
    (880, 160), (880, 240), (880, 320), (880, 400), (880, 480), (880, 560),
    (960, 160), (960, 240), (960, 320), (960, 400), (960, 480), (960, 560),
    (1040, 160), (1040, 240), (1040, 320), (1040, 400), (1040, 480), (1040,560),
    (1120, 160), (1120, 240), (1120, 320), (1120, 400), (1120, 480),(1120,560)],
    dtype="double")
# scan in x axis first (by row, left to right)
image_points_long_scan = np.reshape(np.reshape(image_points, [CHESS_HEIGHT,
                                                              CHESS_WIDTH, 2],
                                               order='fortran'), [ALL_CHESS,1,
                                                                  2])

#scan in reversed 'y' axis (bottom to top, by column)
image_points_short_scan = np.reshape(np.flip(np.reshape(image_points,
                                                        [CHESS_WIDTH,
                                                         CHESS_HEIGHT, 2],
                                                        order='c'), axis =1),
                                     [ALL_CHESS,1,2])


def compute3DPoints(corners, undistorted, registration):
    result = []
    for c in corners:
        xy = c[0]
        x = int(round(xy[0]))
        y = int(round(xy[1]))
        result.append([registration.getPointXYZ(undistorted, y, x)])
    return np.array(result)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def nanToNull(x):
    return None if math.isnan(x) else x

def filterNaN(l):
    return [[nanToNull(x), nanToNull(y), nanToNull(z)] for [x,y,z] in l]


def shortToLongScan(s, isShortScan):
    if isShortScan == True:
        # Returns 3D points,
        # scanned top to bottom, left-to right using the wide side.
        # and serialized as a C array (row-major)
        top2Bot = np.flip(np.reshape(s, [CHESS_WIDTH,CHESS_HEIGHT, 3],
                                     order='c'), axis =1)
        return list(itertools.chain(*np.transpose(top2Bot, (1,0,2)).tolist()))
    else:
        return s

# from  [[[1, 2]], [[2, 3]], [[3, 5]]] to  [[1, 2], [2, 3], [3, 5]]
def peelInnerList(p):
    return filterNaN(list(itertools.chain(*p.tolist())))


def orderCorners(corners):
    distance = np.linalg.norm(corners[1:] -corners[:-1], axis=2)
    shortScan = np.sum(distance[CHESS_HEIGHT-1::CHESS_HEIGHT])/(1.0*
                                                                (CHESS_WIDTH-1))
    longScan = np.sum(distance[CHESS_WIDTH-1::CHESS_WIDTH])/(1.0*
                                                             (CHESS_HEIGHT-1))
    if shortScan > longScan:
        print 'Changing scan order'
        print shortScan, longScan
        #print distance[5::6]
        #print distance[12::13]

        # Short side was the principal side to scan because
        # it found the south-west corner first, and started scanning bottom to
        # top, using the skinny side.
        #
        # Otherwise, it would have scanned top to bottom (left-to right) using
        # the wide side.
        return image_points_short_scan, True
    else:
        return image_points_long_scan, False

def mainSnapshot(options=None):
    pipeline = CudaKdePacketPipeline()
    fn = Freenect2()
    device, listener, registration = initCamera(fn, pipeline)
    warmUp(listener)

    undistorted = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)
    registered = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)
    result = None
    while True:
        frames = listener.waitForNewFrame()
        color = frames["color"]
        depth = frames["depth"]
        registration.apply(color, depth, undistorted, registered)

        # kinect flips X axis
        regArray = registered.asarray(np.uint8)
        regArray = np.flip(regArray, 1)
        imgRGB = cv2.cvtColor(regArray, cv2.COLOR_RGBA2RGB)
        ret, buf = cv2.imencode(".jpg", imgRGB)
        if ret == True:
            data = base64.b64encode(buf)
            result = {'width': FRAME_WIDTH, 'height': FRAME_HEIGHT,
                      'data': data}
            break
        else:
            print 'fail'
        listener.release(frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    shutdownCamera(device);
    return result


def mainCalibrate(options=None):
    counter = 0
    pipeline = CudaKdePacketPipeline()
    fn = Freenect2()
    device, listener, registration = initCamera(fn, pipeline)
    warmUp(listener)
    undistorted = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)
    registered = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)
    result = None
    while True:
       frames = listener.waitForNewFrame()
       color = frames["color"]
       depth = frames["depth"]
       registration.apply(color, depth, undistorted, registered)

       # kinect flips X axis
       regArray = np.flip(registered.asarray(np.uint8), 1)
       flipDepthFrame(undistorted)

       gray = cv2.cvtColor(regArray,cv2.COLOR_RGBA2GRAY)
       ret, corners = cv2.findChessboardCorners(gray, (CHESS_HEIGHT,
                                                       CHESS_WIDTH),None)
       print corners
       if ret == True:
           print 'got it'
           points2D, isShortScan = orderCorners(corners)
           print points2D
           corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1),
                                       criteria)
           points3D = compute3DPoints(corners2, undistorted, registration)
           print points3D
           if (options and options['onlyPoints3D'] == True):
               result = {'points3D':
                         shortToLongScan(peelInnerList(points3D), isShortScan),
                        'points2D': [CHESS_HEIGHT, CHESS_WIDTH]}
               break
           else:
               distortion= np.zeros((4,1))
               ok, rot, trans, inl = cv2.solvePnPRansac(points3D, points2D,
                                                        camera_matrix,
                                                        distortion,
                                                        reprojectionError=2.0,
                                                        iterationsCount=1000,
                                                        flags=cv2
                                                        .SOLVEPNP_ITERATIVE)
               print ok
               if ok == True:
                   print 'Rotation {0}'.format(rot)
                   print 'Translation {0}'.format(trans)
                   print 'Inliers# {0}'. format(len(inl))
                   print 'Checking...'
                   all, _ = cv2.projectPoints(points3D, rot, trans,
                                              camera_matrix, distortion)
                   print all
                   result = {'rotation': list(itertools.chain(*rot.tolist())),
                             'translation': list(itertools.chain(*trans
                                                                 .tolist())),
                             'projMat': openGLProjMat(),
                             'viewMat': openGLViewMat(rot, trans),
                             'points2D': [CHESS_HEIGHT, CHESS_WIDTH],
                             'points3D':
                             shortToLongScan(peelInnerList(points3D),
                                             isShortScan)}
                   break

#           img = cv2.drawChessboardCorners(regArray, (6, 13), corners2, ret)
#           cv2.imshow('corners', img)
       else:
           print 'nope'

#       cv2.imshow("depth", depth.asarray() / 4500.)
#       cv2.imshow("registered", regArray)

       listener.release(frames)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    shutdownCamera(device);
    return result


def mainCalibrateProjector(objPoints, imagePoints, shape):
    print objPoints, imagePoints, shape
    objPoints = np.array([np.array(xi, dtype=np.float32) for xi in objPoints])
    imagePoints = np.array([np.array(xi, dtype=np.float32) for xi in imagePoints])
    shape = (shape[0], shape[1])
    print 'shape', shape
    fl = cv2.CALIB_USE_INTRINSIC_GUESS
    retval, cameraMatrix, distCoeffs, _, _ = cv2.calibrateCamera(objPoints,
                                                                 imagePoints,
                                                                 shape,
                                                                 camera_matrix,
                                                                 None,
                                                                 flags=fl)
    print retval, cameraMatrix, distCoeffs
    return {'retval': retval,
            'cameraMatrix': list(itertools.chain(*cameraMatrix.tolist())),
            'distCoeffs': list(itertools.chain(*distCoeffs.tolist()))}

if __name__ == "__main__":
    mainCalibrate()
