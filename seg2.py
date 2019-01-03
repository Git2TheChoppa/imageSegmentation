import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from skimage import segmentation
from skimage import filters
from skimage.feature import canny

coins = cv2.imread('quarters.jpg')
coins = cv2.cvtColor(coins, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(coins, cv2.COLOR_RGB2GRAY)
plt.imshow(gray)

traffic = cv2.imread('traffic.jpg')
traffic = cv2.cvtColor(traffic, cv2.COLOR_BGR2RGB)
gray2 = cv2.cvtColor(traffic, cv2.COLOR_RGB2GRAY)
plt.imshow(gray2)
'''
mask = gray >= filters.threshold_yen(gray)
plt.imshow(mask)
thresh = filters.threshold_isodata(coins)
binary = gray > thresh
plt.imshow(binary)
#clean_border = segmentation.clear_border(mask).astype(np.int)
#plt.imshow(clean_border)
'''

edges = canny(gray, sigma=5, low_threshold=10, high_threshold=60)
plt.imshow(edges)

coin_edges = segmentation.mark_boundaries(gray, edges, color = (206, 93, 33), outline_color = (206, 93, 33)) # a nice blue color
plt.imshow(coin_edges)
plt.show()


edges2 = canny(gray2, sigma = 5, low_threshold = 10, high_threshold = 60)
plt.imshow(edges2)

car_edges = segmentation.mark_boundaries(traffic, edges2, color = (205, 90, 30), outline_color = (205, 90, 30))
plt.imshow(car_edges)
plt.show()


#cv2.imshow('CARS', car_edges)
cv2.imwrite('cars2.png', car_edges)



from PIL import Image


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open('Blue_Output_Traffic.png')
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()
 
 
if __name__ == '__main__':
    image = 'Blue_Output_Traffic.jpg'
    crop(image, (0, 200, 1200, 800), 'cropped.png')
    
    
    
    
    
    
