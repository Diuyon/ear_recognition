import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.segmentation import active_contour
import scipy

def alpha_parm(x):
    global alpha
    alpha = x

def betha_parm(x):
    global betha
    betha = x

def gamma_parm(x):
    global gamma
    gamma = x

def line_parm(x):
    global line
    line = x
#   print(alpha*0.001)

def edge_parm(x):
    global edge
    edge = x    

def ellip_x(x):
    global edge
    edge = x        

colors = (0,255,0)

cv2.namedWindow("Original",cv2.WINDOW_NORMAL)


name = "ear2.jpg"
image_col = cv2.imread(name)
image = cv2.imread(name,0)


blur = cv2.GaussianBlur(image,(9,9),0)

alpha = 70
betha = 170
gamma= 10
line= 150
edge = 750

cv2.resizeWindow('Original', 600,600)
cv2.createTrackbar("Alpha", 'Original', alpha, 1000, alpha_parm)
cv2.createTrackbar("Betha", 'Original', betha, 1000, betha_parm)
cv2.createTrackbar("Gamma", 'Original', gamma, 1000, gamma_parm)
cv2.createTrackbar("Line", 'Original', line, 1000, line_parm)
cv2.createTrackbar("Edge", 'Original', edge, 1000, edge_parm)

tmp = (alpha,betha,gamma,line,edge)
while True:
    s = np.linspace(0, 2*np.pi, 400)
    x = 110 + 65*np.cos(s)
    y = 100 + 100*np.sin(s)
    init = np.array([x, y]).T
    
    
    #alpha 0.07
    #beta 0.17
    #w_line 0.15
    #w_edge=-0.75
    #gamma=0.01
    
    snake = active_contour(blur,init,alpha*0.001, betha*0.001, line*0.001, edge*(-0.001), gamma*0.001, max_iterations=200)
    snake = np.expand_dims(snake, axis=0)
    
    
    snake = np.rint(snake)
    snake = snake.astype(int)
    
    init = np.expand_dims(init, axis=0)
    init = np.rint(init)
    init = init.astype(int)
    
    cv2.polylines(image_col,snake, True, color=colors, thickness=2  )
    cv2.polylines(image_col,init, True, color=(255,0,0), thickness=2)

    cv2.imshow("Original",image_col)
    
    print((alpha,betha,gamma,line,edge))
    if (alpha,betha,gamma,line,edge) != tmp:
        image_col = cv2.imread(name)



    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()

