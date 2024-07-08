from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
from PIL import Image

output_img_size = 224  # Square. 

# If required, create a face detection pipeline using MTCNN:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Model. 
mtcnn = MTCNN(image_size=output_img_size, margin=0, selection_method="probability", device=device)

# Load video. 
cap = cv2.VideoCapture('/home/alex/바탕화면/Data/batch1/4-30-1920x1080.mp4')
i = 0

# img = cv2.imread("/home/alex/바탕화면/AU/test_angle/pitch.jpg")
# img_crop = mtcnn(img, save_path="output.jpg")

while (cap.isOpened()): 
    
    ret, frame = cap.read()
    if not ret: 
        break
    
    # Built for PIL image (RGB order assumed).
    # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLORBGR2RGB))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect. 
    bboxes,_ = mtcnn.detect(rgb_frame)
    
    x_min = int(bboxes[0][0])
    y_min = int(bboxes[0][1])
    x_max = int(bboxes[0][2])
    y_max = int(bboxes[0][3])
    
    # Crop face. 
    face_cropped = frame[y_min:y_max, x_min:x_max]
    
    # If cropped face is smaller than 224 x 224: 
    cropped_area = int(face_cropped.shape[0] * face_cropped.shape[1])
    if cropped_area < int(224 * 224): 
        face_resized = cv2.resize(face_cropped, (224,224), interpolation=cv2.INTER_CUBIC)  # Enlarge.
    # Else: 
    else: 
        face_resized = cv2.resize(face_cropped, (224,224), interpolation=cv2.INTER_AREA)  # Shrink.
    
    # save image
    name = f"output/{i}.jpg"
    cv2.imwrite(name, face_resized)

    # Cropped face. 
    i += 1 
    if i == 100: 
        break

cap.release()