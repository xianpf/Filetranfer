import numpy as np 
import cv2

# a = cv2.imread('image/tiananmen1.png')
# b = cv2.cvtColor(a, cv2.COLOR_BGR2YUV)
# b[:,:,0] = cv2.equalizeHist(b[:,:,0])
# c = cv2.cvtColor(b, cv2.COLOR_YUV2BGR)
# cv2.imwrite('image/equal.png',c)
# cv2.imshow('bc',c)
# cv2.waitKey(0)

a = cv2.imread('image/test.jpg')
b = np.zeros(a.shape)
for i in range(3):
    b[:,:,i] = cv2.equalizeHist(a[:,:,i])
cv2.imwrite('image/equal.jpg', b)
cv2.imshow('equal', b)
cv2.waitKey(0)


