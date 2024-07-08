import numpy as np
import matplotlib.pyplot as plt
import cv2

raw_value = []
with open('model.txt') as file:
    for line in file:
        raw_value.append(line)
model_points = np.array(raw_value, dtype=np.float32)
model_points = np.reshape(model_points, (3, -1)).T

# model_points[:, [1, 2]] = model_points[:, [2, 1]]
model_points[:, 2] *= -1

# model_points = np.array([
#                             (0.0, 0.0, 0.0),             # Nose tip
#                             (0.0, -330.0, -65.0),        # Chin
#                             (-165.0, 170.0, -135.0),     # Left eye left corner
#                             (165.0, 170.0, -135.0),      # Right eye right corne
#                             (-150.0, -150.0, -125.0),    # Left Mouth corner
#                             (150.0, -150.0, -125.0)      # Right mouth corner                         
#                         ])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter3D(model_points[:,0], model_points[:,1], model_points[:,2], 'gray')
plt.show()