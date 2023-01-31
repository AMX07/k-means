import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image into variable 'img'
# #please note that the images shown in video are in
# descending order of the quality of segmentation that is done on them, therefore last image is the one with best results
# as per the criteria mentioned
img = cv2.imread('image1.jpg')

# convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = img.reshape((-1, 3))

# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0) #no. of iternations and epislon value

# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(img.shape)


# show the image
plt.imshow(segmented_image)
plt.show()

