import logging
logging.basicConfig(level=logging.DEBUG)
import cProfile
import time
import numpy as np
import cv2
import sys
import json

from .util import flipDepthFrame

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import CudaKdePacketPipeline

import bodyparts as bp

MAX_FLOAT32 = 2.0**31 - 1

DISTANCE_THRESHOLD = 50.

MAX_DISTANCE = 8000.
MIN_DISTANCE = 500.

MIN_CONTOUR_AREA = 1000
MIN_HOLE_AREA = 300

FRAME_WIDTH = 512
FRAME_HEIGHT = 424

def scale(inMat):
    a = 255.0 /(MAX_DISTANCE-MIN_DISTANCE)
    b = -a * MIN_DISTANCE
    scaled = np.clip(inMat, MIN_DISTANCE, MAX_DISTANCE) * a + b
    return scaled.astype(np.uint8)

def unscale(inMat):
    a = (MAX_DISTANCE-500.0)/255.0
    b = MIN_DISTANCE
    unscaled = inMat * a + b
    unscaled = unscaled.astype(np.float32)
    unscaled[unscaled == MAX_DISTANCE] = MAX_FLOAT32
    return unscaled

def initCamera(fn, pipeline):
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No device connected!")
        sys.exit(1)
    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)
    types = (FrameType.Ir | FrameType.Depth)
    listener = SyncMultiFrameListener(types)
    device.setIrAndDepthFrameListener(listener)
    device.startStreams(rgb= False, depth=True)
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())
    return (device, listener, registration)

def paintContour(contours, hierarchy, minVal):
    maxArea = 0.;
    c = -1
    for idx, con in enumerate(contours):
        area = cv2.contourArea(con)
        if area > MIN_CONTOUR_AREA:
            c = idx if area > maxArea else c
            maxArea = area if area > maxArea else maxArea

    #  Draw only the biggest and its holes
    alpha = np.zeros(minVal.shape, dtype=np.uint8)
    if c >= 0:
        cv2.drawContours(alpha, [contours[c]], 0, 255, -1)
        for index, child in  enumerate(hierarchy[0]):
            if child[3] == c:
                if cv2.contourArea(contours[index]) > MIN_HOLE_AREA:
                    #print 'hole', index, c
                    cv2.drawContours(alpha, [contours[index]], 0, 0, -1)
    return alpha

#background subtraction
def subtraction(minVal, d):
    d[d == 0.] = MAX_FLOAT32
    #cv2.imshow('Before subtraction', d / 4500.)
    #         cv2.waitKey(1)
    #foregroundSize = d.size - d[d>=(minVal-50.)].size
    d[d>=(minVal-DISTANCE_THRESHOLD)] = MAX_FLOAT32
    #print count, foregroundSize
    #cv2.imshow('After subtraction', d / 4500.)

    ret, thr = cv2.threshold(scale(d), 200, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('Threshold', thr)
    _, contours, hierarchy = cv2.findContours(thr, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_NONE)
    return paintContour(contours, hierarchy, minVal)


def warmUp(listener):
    count = 0
    while count < 50:
        frames = listener.waitForNewFrame()
        count = count + 1
        listener.release(frames)

def inpaint(minVal):
    #inpainting the background
    mask = np.zeros(minVal.shape, dtype=np.uint8)
    mask[minVal == MAX_FLOAT32] = 255
#    cv2.imshow('before', minVal / 4500.)
#    cv2.waitKey(0)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    scaledMin = scale(minVal)
    scaledPatchedMin = cv2.inpaint(scaledMin,mask,3,cv2.INPAINT_TELEA)
    approxMinVal = unscale(scaledPatchedMin)
    #print (approxMinVal[mask == 255])
    minVal[mask == 255] = approxMinVal[mask == 255]
#    cv2.imshow('after', minVal / 4500.)
#    cv2.waitKey(0)
    return minVal

def computeBackground(listener, registration):
    count = 0
    undistorted = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)
    minVal = np.full((FRAME_HEIGHT, FRAME_WIDTH), MAX_FLOAT32, dtype=np.float32)
    while count < 60:
        frames = listener.waitForNewFrame()
        depth = frames["depth"]
        registration.undistortDepth(depth, undistorted)
        # kinect flips X axis
        flipDepthFrame(undistorted)

        d =  undistorted.asarray(np.float32)
        zeros = d.size - np.count_nonzero(d)
        #print('Input:zeros:' + str(zeros) + ' total:' + str(d.size))
        d[d == 0.] = MAX_FLOAT32
        minVal = np.minimum(minVal, d)
        #print ('Minval: zeros:' + str(minVal[minVal == MAX_FLOAT32].size) +
        #       ' total:' + str(minVal.size))
        count = count + 1
        listener.release(frames)
    return inpaint(minVal)

def shutdownCamera(device):
    device.stop()
    device.close()

def mainSegment(options = None):
    print '--------------------------------------------'
    print options
    counter = 0
    pipeline = CudaKdePacketPipeline()
    fn = Freenect2()
    device, listener, registration = initCamera(fn, pipeline)
    warmUp(listener)
    minVal = computeBackground(listener, registration)
    undistorted = Frame(FRAME_WIDTH, FRAME_HEIGHT, 4)
    net = bp.newNet()
    t0 =  time.clock()
    counterOld = counter
    while True:
        frames = listener.waitForNewFrame()
        depth = frames["depth"]
        registration.undistortDepth(depth, undistorted)
        # kinect flips X axis
        flipDepthFrame(undistorted)

        d =  np.copy(undistorted.asarray(np.float32))
        alpha = subtraction(minVal, d)
        if (options and options.get('display')):
            cv2.imshow('Contour', alpha)

        result = bp.process(options, net, scale(d), alpha, registration,
                            undistorted, device)

        if (options and options.get('display')):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        listener.release(frames)
        counter = counter + 1
        if counter % 10 == 0:
            t1 = time.clock()
            print '{:.3f} images/sec'.format((counter - counterOld)/(t1-t0))
            t0 = t1
            counterOld = counter
        #print(result)
        yield json.dumps(result)
    shutdownCamera(device);

def loop(options = None):
    g = mainSegment(options)
    for res in g:
        print (res)

if __name__ == "__main__":
    cProfile.run('mainSegment()') #mainSegment()
