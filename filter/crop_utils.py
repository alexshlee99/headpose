import numpy as np
from PIL import Image
import math
import cv2


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def point_rotation(prev, center, angle): 
    
    new_x = (prev[0] - center[0]) * math.cos(angle) - (prev[1] - center[1]) * math.sin(angle) + center[0]
    new_y = (prev[0] - center[0]) * math.sin(angle) - (prev[1] - center[1]) * math.cos(angle) + center[1]

    return int(new_x), int(new_y)
    

def alignment_procedure(img, left_eye, right_eye, nose):
    """
    Align face by calculating rotation from landmarks (only accounts for roll-rotation!)
    """
    
    #this function aligns given face in img based on left and right eye coordinates

    #left eye is the eye appearing on the left (right eye of the person)
    #left top point is (0, 0)

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    #-----------------------
    # Get center coordinates between eyes. 
    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))
    
    if False:
        img = cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 2, (0, 255, 255), 2)
        img = cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 2, (255, 0, 0), 2)
        img = cv2.circle(img, center_eyes, 2, (0, 0, 255), 2)
        img = cv2.circle(img, (int(nose[0]), int(nose[1])), 2, (255, 255, 255), 2)
    
    
    # Compare with nose x and y. 
    # if 
    
    
    #-----------------------
    # Find rotation direction. 

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y) 
        direction = 1 #rotate inverse direction of clock 
    
    #-----------------------
    # Find length of triangle edges
    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    #-----------------------
    # Apply cosine rule
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)  # Why use cosine rule, when right triangle???
        
        #PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        #In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))
        
    
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #-----------------------
        #rotate base image
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle, expand=True))  # center=left_eye, expand=True


        # # Customized: 
        # # Rotate, then take equal same length from nose!
        # bbox_x_min = bbox_coords[0]
        # bbox_x_max = bbox_coords[2]
        # bbox_y_min = bbox_coords[1]
        # bbox_y_max = bbox_coords[3]
        
        # tl = [bbox_x_min, bbox_y_min]
        # tr = [bbox_x_max, bbox_y_min]
        # bl = [bbox_x_min, bbox_y_max]
        # br = [bbox_x_max, bbox_y_max]

        # # By rotating with PIL, we lose information (since rotated output img dim. == input image dim.). 
        # # With expand, we don't lose information? 
        # new_tl = point_rotation(tl, left_eye, angle)
        # new_tr = point_rotation(tr, left_eye, angle)
        # new_bl = point_rotation(bl, left_eye, angle)
        # new_br = point_rotation(br, left_eye, angle)
        
        # img = img[new_tl[1]:new_br[1], new_tl[0]:new_br[0]]

    #-----------------------

    return img #return img anyway