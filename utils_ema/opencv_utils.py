import cv2

def rot2rvec(rmat):
    rvec, _ = cv2.Rodrigues(rmat)

def rvec2rot(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
