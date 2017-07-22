import cv2
import numpy as np
import sys
import caffe
import math
import os

# Models from "Real-Time Human Motion Capture with Multiple Depth Camera"
#   by Alireza Shafaei and  James J. Little
#     UBC, Vancouver, Canada
# https://github.com/ashafaei/dense-depth-body-parts.git

MODEL_PROTO='deploy.prototxt'
MODEL_WEIGHTS='hardpose_69k.caffemodel'

KP_SIZE = 10
BACKGROUND = [255, 255, 255]
COLOR_MAP = np.array([
    [255, 106, 0],
    [255, 0, 0],
    [255, 178, 127],
    [255, 127, 127],
    [182, 255, 0],
    [218, 255, 127],
    [255, 216, 0],
    [255, 233, 127],
    [0, 148, 255],
    [72, 0, 255],
    [48, 48, 48],
    [76, 255, 0],
    [0, 255, 33],
    [0, 255, 255],
    [0, 255, 144],
    [178, 0, 255],
    [127, 116, 63],
    [127, 63, 63],
    [127, 201, 255],
    [127, 255, 255],
    [165, 255, 127],
    [127, 255, 197],
    [214, 127, 255],
    [161, 127, 255],
    [107, 63, 127],
    [63, 73, 127],
    [63, 127, 127],
    [109, 127, 63],
    [255, 127, 237],
    [127, 63, 118],
    [0, 74, 127],
    [255, 0, 110],
    [0, 127, 70],
    [127, 0, 0],
    [33, 0, 127],
    [127, 0, 55],
    [38, 127, 0],
    [127, 51, 0],
    [64, 64, 64],
    [73, 73, 73],
    [0, 0, 0],
    [191, 168, 247],
    [192, 192, 192],
    [127, 63, 63],
    [127, 116, 63],
    BACKGROUND
]);

COMPRESS_FRONT = [
#head
    (41, 38),
    (42, 38),
    (40, 38),
    (39, 38),
#neck
    (10, 9),
    (8, 9),
    (16, 9),
    (15, 9),
    (17, 9),
#shoulders (assume front)
    (14, 12),
    (13, 11),
#body (assume front)
    (6, 0),
    (4, 1),
    (7, 2),
    (5, 3),
#feet (assume front)
    (25, 26),
    (24, 27),
];

BIMODAL_SET = set([
    28,33,32,31,30,
    34,35,36,37,29,
    18,19,20,21,22,23
])

BACKGROUND_INDEX = COLOR_MAP.shape[0] - 1

def compressVector():
    res = np.arange(COLOR_MAP.shape[0], dtype=np.uint8)
    for index, val in COMPRESS_FRONT:
        res[index] = val
    return res

COMPRESS_VECTOR = compressVector()

INNER_WINDOW = 190
MARGIN = 30
WINDOW_SIZE = INNER_WINDOW + 2*MARGIN
NUM_BLOCKS = 10
SIZE_BLOCK = (INNER_WINDOW / NUM_BLOCKS)

#ignore body part if there are less than MIN_COUNT pixels
MIN_COUNT = 400

#returns [left,right) and [top, bottom) (not including `right` and `bottom`)
def boundaries(alpha):
    psum_width = alpha.sum(axis=0)
    ind = psum_width.nonzero()
    if ind[0].size == 0:
        return (False, ())
    else:
        left = ind[0][0]
        right = ind[0][-1]
        psum_height = alpha.sum(axis=1)
        if ind[0].size == 0:
            return (False, ())
        else:
            ind = psum_height.nonzero()
            top = ind[0][0]
            bottom = ind[0][-1]
            return (True, (left, right+1, top, bottom+1))

#respect aspect ratio
def newDimensions(height, width):
    #print height, width
    if height > width:
        newHeight = INNER_WINDOW
        newWidth = int(round(width * (newHeight/ float(height))))
    else:
        newWidth = INNER_WINDOW
        newHeight =  int(round(height * (newWidth/ float(width))))
    return (newHeight, newWidth)

def crop(data, alpha, box):
    left, right, top, bottom = box
    width = right-left;
    height = bottom-top;
    newHeight, newWidth = newDimensions(height, width)
    data = data[top:bottom, left:right]
    alpha = alpha[top:bottom, left:right]
    return (data, alpha, newWidth, newHeight)

def resize(data, alpha, newWidth, newHeight):
    newData = cv2.resize(data, (newWidth, newHeight),
                         interpolation = cv2.INTER_LANCZOS4)
    newAlpha = cv2.resize(alpha, (newWidth, newHeight),
                          interpolation = cv2.INTER_NEAREST)
    return (newData, newAlpha)


# output is float [0, 1.0] with background=1.0
def normalizeData(data, alpha):
    meanDepth = data[alpha == 255].mean()
    # note that 55 is 1.6m + 0.5m offset in a range [0.5, 8.0]...
    data[alpha == 255] = np.clip(data[alpha == 255] +
                                 (55.0 - meanDepth), 0,
                                 255).astype(np.float32)
    data[alpha != 255] = 255.
    data = data /255.0
    return data

def place(data, alpha, newWidth, newHeight):
    finalData = np.ones((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    finalAlpha = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
    locH = int(np.floor((WINDOW_SIZE - newHeight)/2.))+1
    locW = int(np.floor((WINDOW_SIZE - newWidth)/2.))+1
    finalData[locH:locH+newHeight, locW:locW+newWidth] = data
    finalAlpha[locH:locH+newHeight, locW:locW+newWidth] = alpha
    return finalData, finalAlpha, locW, locH

#use tiling to stride inside the cache
def blockArgmax(mat):
    # input is [46,INNER_WINDOW , INNER_WINDOW] np.float
    #output is [WINDOW_SIZE, WINDOW_SIZE] np.uint8
    res = np.ones((WINDOW_SIZE, WINDOW_SIZE), dtype = np.uint8)*BACKGROUND_INDEX
    for rB in range(NUM_BLOCKS):
        for rC in range(NUM_BLOCKS):
            blMat = mat[:,rB*SIZE_BLOCK:(rB+1)*SIZE_BLOCK,
                        rC*SIZE_BLOCK:(rC+1)*SIZE_BLOCK]
            s = np.argmax(np.copy(np.transpose(blMat, (1,2,0)),order='C'),
                          axis=2).astype(np.uint8)
            s = COMPRESS_VECTOR[s]

            res[MARGIN+rB*SIZE_BLOCK:MARGIN+(rB+1)*SIZE_BLOCK,
                MARGIN+rC*SIZE_BLOCK:MARGIN+(rC+1)*SIZE_BLOCK] = s
    return res

# resolve left/right confusion by ignoring the minority ones in a bimodal dist
def filterLeftRight(mat):
    c = mat[:,1]
    maxC = np.max(c)
    minC = np.min(c)
    a = 255.0/(maxC-minC)
    b = -a*minC
    newC = (a*c+b+0.5).astype(dtype = np.uint8)
    thr, thMat = cv2.threshold(newC, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    nonzeros = np.count_nonzero(thMat)
    zeros = thMat.size - nonzeros
    mapThr = int(0.5 + (thr - b)/a)
    mask = mat[:,1] > mapThr
    return mat[mask] if (nonzeros > zeros) else  mat[np.logical_not(mask)]

def analyze(mat, box, registration, undistorted):
    # input is [FRAME_HEIGHT, FRAME_WIDTH] np.uint8
    #output is [(part, (mean_X, mean_Y, mean_Z))]
    left, right, top, bottom = box
    target =  mat[top:bottom, left:right]
    values, counts = np.unique(target, return_counts=True)
    pairs = zip(values, counts)
    pairs = [(x,y) for (x, y) in pairs if y>MIN_COUNT and x != BACKGROUND_INDEX]
    result = []

    for part, _ in pairs:
        #inefficient in general, but typically just a few body parts, e.g., <10
        all = np.argwhere(target == part) + np.array([top, left])
        if (part in BIMODAL_SET):
            all = filterLeftRight(all)
        points3D_X = 0.0
        points3D_Y = 0.0
        points3D_Z = 0.0
        count = 0
        allList = all.tolist()
        for r, c in allList:
            x, y, z = registration.getPointXYZ(undistorted, r, c)
            if not math.isnan(x):
                points3D_X = points3D_X + x
                points3D_Y = points3D_Y + y
                points3D_Z = points3D_Z + z
                count = count + 1
        centroid = (points3D_X/count, points3D_Y/count, points3D_Z/count)
        result.append((int(part), centroid))

    return result


def inference(net, data, alpha):
    net.blobs['data'].data[...] = data
    output = net.forward()
    output_prob = np.copy(output['prob'][0]) #slow to reference caffe blobs
    #    denseSmall = np.argmax(output_prob, axis=0)
    dense = blockArgmax(output_prob)
#    dense = np.zeros(((WINDOW_SIZE, WINDOW_SIZE), dtype=np.uint8)
#    print output['prob'].shape,  output['prob'][0]
#    dense =  output['prob'][0][0]
    dense[alpha == 0] = BACKGROUND_INDEX
    return dense

def newNet():
    script_dir = os.path.dirname(__file__)
    caffe.set_mode_gpu()
    return caffe.Net(os.path.join(script_dir, MODEL_PROTO),
                     os.path.join(script_dir, MODEL_WEIGHTS), caffe.TEST)

def toColorImage(result, projectedInfo):
    keyPoints = [cv2.KeyPoint(p[0], p[1], KP_SIZE) for (x,p) in projectedInfo]
    newImage = COLOR_MAP[result].astype(np.uint8)  # / 255.
    # RGB to BGR
    img =  newImage[:,:,::-1]
    return cv2.drawKeypoints(img, keyPoints, None, color = (0, 0, 0))

def undoPlace(result, placeLoc, afterScaleDim):
    locW, locH = placeLoc
    width, height = afterScaleDim
    return result[locH:locH+height, locW:locW+width]

def undoResize(result, box):
    left, right, top, bottom = box
    width = right-left;
    height = bottom-top;
    return cv2.resize(result, (width, height),
                      interpolation = cv2.INTER_NEAREST)

def undoCrop(result, box, originalDim):
    left, right, top, bottom = box
    width, height = originalDim
    big = np.ones((height, width), dtype = np.uint8) * BACKGROUND_INDEX
    big[top:bottom, left:right] = result
    return big

def undoTransforms(result, placeLoc, afterScaleDim, box, originalDim):
    result = undoPlace(result, placeLoc, afterScaleDim)
    result = undoResize(result, box)
    return undoCrop(result, box, originalDim)

def project(info, device):
    depth = device.getIrCameraParams()
    camera_matrix = np.array([[depth.fx, 0, depth.cx],
                              [0, depth.fy, depth.cy],
                              [0, 0, 1]], dtype = np.float)
    def projectOne(p):
        x, y, z = p
        u, v, l =  np.matmul(camera_matrix, np.array([[x],[y],[z]])).tolist()
        u = int(round(u[0] / l[0]))
        v = int(round(v[0] / l[0]))
        return (u,v)

    return map(lambda (t,p): (int(t), projectOne(p)), info)

def process(options, net, data, alpha, registration, undistorted, device):
    height, width = data.shape
    #print width, height
    #print alpha.shape
    ok, box = boundaries(alpha)
    #print ok, box
    if ok == True:
        data, alpha, newWidth, newHeight = crop(data, alpha, box)
        data, alpha = resize(data, alpha, newWidth, newHeight)
        data = normalizeData(data, alpha)
        data, alpha, locW, locH = place(data, alpha, newWidth, newHeight)
        result = inference(net, data, alpha)

        big = undoTransforms(result, (locW, locH), (newWidth, newHeight), box,
                             (width, height))
        info = analyze(big, box, registration, undistorted)
        projectedInfo = project(info, device)

        if (options and options.get('display')):
            cv2.imshow('bigOutput', toColorImage(big, projectedInfo))

        return (info, projectedInfo)
    else:
        return None
