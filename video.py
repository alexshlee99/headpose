import cv2 
import os


def extract_integer(filename):
    return int(filename.split('.')[0])

imgs_path = 'output'

w = 640
h = 480
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

for img_name in sorted(os.listdir(imgs_path), key=extract_integer):
    
    img_path = os.path.join('output', img_name)
    
    print(img_path)
    
    img = cv2.imread(img_path)
    
    out.write(img)

out.release()
cv2.destroyAllWindows()