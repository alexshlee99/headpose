import cv2 
import numpy as np

kist_img = cv2.imread('/home/alex/바탕화면/Data/AffWild2/result/1-30-1280x720/00033.jpg')
aff_img = cv2.imread('/home/alex/다운로드/cropped_aligned/1-30-1280x720/00033.jpg')
aff_img = cv2.resize(aff_img, (224, 224))

comparison = np.hstack((kist_img, aff_img))
cv2.imshow('test', comparison)
cv2.waitKey()
