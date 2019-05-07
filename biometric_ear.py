import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.segmentation import active_contour
import scipy



def CannyThreshold(val):
    global low_threshold
    low_threshold = val


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
    global ax_x
    ax_x = x    
def alt_y(x):
    global altura
    altura = x         
def tresh_hold (x):
    global th
    th = x      
def dilatation (x):
    global min_size
    min_size = x      
    
colors = (0,255,0)

cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
cv2.namedWindow("Masked",cv2.WINDOW_NORMAL)
cv2.namedWindow("Avg_Masked",cv2.WINDOW_NORMAL)
cv2.namedWindow("Treshhold",cv2.WINDOW_NORMAL)
cv2.namedWindow("Canny",cv2.WINDOW_NORMAL)


name = "ear_example.jpg"
#name = "/home/jose/Downloads/ear/raw/051_1.bmp"
origin = cv2.imread(name)

image_col = cv2.imread(name)

image = cv2.imread(name,0)


blur = cv2.GaussianBlur(image_col,(9,9),0)

alpha = 207
betha = 635
gamma= 1
line= 100
edge = 808
ax_x = 111
altura = 122
th=73 #treshold
low_threshold = 100
iterations = 100
min_size = 50


cv2.resizeWindow('Original', 600,600)
cv2.resizeWindow('Masked', 600,600)
cv2.resizeWindow('Avg_Masked', 600,600)
cv2.resizeWindow('Treshhold', 600,600)
cv2.resizeWindow('Canny', 600,600)


cv2.createTrackbar("Alpha", 'Original', alpha, 1000, alpha_parm)
cv2.createTrackbar("Betha", 'Original', betha, 1000, betha_parm)
cv2.createTrackbar("Gamma", 'Original', gamma, 1000, gamma_parm)
cv2.createTrackbar("Line", 'Original', line, 1000, line_parm)
cv2.createTrackbar("Edge", 'Original', edge, 1000, edge_parm)
cv2.createTrackbar("X axis", 'Original', ax_x, 1000, ellip_x)
cv2.createTrackbar("ALtura", 'Original', altura, 1000, alt_y)

cv2.createTrackbar("Tresh", 'Treshhold', th, 1000, tresh_hold)

cv2.createTrackbar("Tresh_max", 'Canny', low_threshold, 1000, CannyThreshold)
cv2.createTrackbar("Min", 'Canny', min_size, 1000, dilatation)


tmp = (alpha,betha,gamma,line,edge,ax_x,altura)
while True:
    s = np.linspace(0, 2*np.pi, 400)
    x = ax_x  + 65*np.cos(s)
    y = 100 + altura*np.sin(s)
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
    
    cv2.polylines(image_col,snake, True, color=colors, thickness=1  )
    cv2.polylines(image_col,init, True, color=(255,0,0), thickness=2)

    # mask defaulting to black for 3-channel and transparent for 4-channel
# (of course replace corners with yours)
    mask = np.zeros(blur.shape, dtype=np.uint8)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = blur.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, snake, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    
    # apply the mask
    masked_image = cv2.bitwise_and(origin, mask)

    avg_masked = cv2.medianBlur(masked_image,5)
    
    tr,thresh1 = cv2.threshold(avg_masked,th,255,cv2.THRESH_BINARY)

    img_blur = cv2.blur(thresh1, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*4, 3)
    mask_1 = detected_edges != 0
    dst = thresh1 * (mask_1[:,:,None].astype(thresh1.dtype))
    dilate = cv2.dilate(dst,(3,3),1000)
    
    dilate = cv2.cvtColor(dilate,cv2.COLOR_RGB2GRAY)
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilate, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    
    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, (3,3))

    
    
    cv2.imshow("Original",image_col)
    cv2.imshow("Masked",masked_image)
    cv2.imshow("Avg_Masked",avg_masked)
    cv2.imshow("Treshhold",thresh1)
    cv2.imshow("Canny",dilate)
    cv2.imshow("Canny",img2)
    cv2.imshow("ss",closing)
     

    
    print((alpha,betha,gamma,line,edge))
    if (alpha,betha,gamma,line,edge,ax_x,altura) != tmp:
        image_col = cv2.imread(name)



    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()

