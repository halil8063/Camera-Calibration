import cv2
import glob
import numpy as np
import matplotlib.image as mp

fname = "calibration*.jpg"
nx = 9
ny = 6
images = glob.glob(fname)
objpoints = []
imgpoints = []
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for fname in images:

    img = mp.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(nx,ny))

    if ret:
      imgpoints.append(corners)
      objpoints.append(objp)

      img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

def calib(img,objpoints,imgpoints):
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist


cam = cv2.VideoCapture(0)

while True:
    ret1,cam1 = cam.read()
    undistorted = calib(cam1,objpoints,imgpoints)

    cv2.imshow("original",cam1)
    cv2.imshow("undistorted",undistorted)
    if cv2.waitKey(25)&0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
