import torch
from torchvision import transforms as trans

### Configuration for Face Recognition. 
data_path = "path/to/data"
work_path = "path/to/work_path"
model_path = work_path/'models'
log_path = work_path/'log'
save_path = work_path/'save'
input_size = [112,112]
embedding_size = 512
use_mobilfacenet = False
facebank_path = data_path/'facebank'
net_depth = 50
drop_ratio = 0.6
net_mode = 'ir_se' # or 'ir'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
data_mode = 'emore'
vgg_folder = data_path/'faces_vgg_112x112'
ms1m_folder = data_path/'faces_ms1m_112x112'
emore_folder = data_path/'faces_emore'
batch_size = 100 # irse net depth 50 
#   batch_size = 200 # mobilefacenet
facebank_path = data_path/'facebank'
threshold = 1.5
face_limit = 10 
#when inference, at maximum detect 10 faces in one image,
min_face_size = 35
# the larger this value, the faster deduction, comes with tradeoff in small faces


### Landmark index (68).
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]