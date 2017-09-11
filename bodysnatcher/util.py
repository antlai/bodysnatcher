import numpy as np

#Kinect flips the X axis of all streams
def flipDepthFrame(d):
    dArray = d.asarray(np.float32)
    for r in range(dArray.shape[0]):
        dArray[r][:] = dArray[r][::-1].copy()
