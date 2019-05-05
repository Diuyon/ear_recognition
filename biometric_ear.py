import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.segmentation import active_contour
import scipy

cv2.namedWindow('Original',cv2.WINDOW_NORMAL)


name = "ear_example.jpg"
image_col = cv2.imread(name)
image = cv2.imread(name,0)


blur = cv2.GaussianBlur(image,(9,9),0)



s = np.linspace(0, 2*np.pi, 400)
x = 110 + 65*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([x, y]).T


snake = active_contour(blur,init,alpha=0.07, beta=0.17, w_line=0.15, w_edge=-0.75, gamma=0.01, max_iterations=200)
snake = np.expand_dims(snake, axis=0)



colors = (0,255,0)
snake = np.rint(snake)
snake = snake.astype(int)

init = np.expand_dims(init, axis=0)
init = np.rint(init)
init = init.astype(int)

cv2.polylines(image_col,snake, True, color=colors, thickness=2  )
cv2.polylines(image_col,init, True, color=(255,0,0), thickness=2,)


cv2.resizeWindow('Original', 600,600)

cv2.imshow("Original",image_col)


