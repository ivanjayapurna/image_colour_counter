import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time
import cv2
import csv


##############
### INPUTS ###
##############

# number of colours to cluster
n_colors = 6

# colour input rgb's
# INPUT NAMES AND RGB VALUES (note: leave black and white by default)
# imagecolorpicker: https://imagecolorpicker.com
use_color_picking = True
c1 = [0,0,0]
c2 = [255,255,255]
c3 = [108, 44, 42]
c4 = [49, 52, 180]
c5 = [139, 77, 108]
c6 = [52, 32, 45]
colour_clusters = np.array([c1,c2,c3,c4,c5,c6], np.float64)

# Load image
img_name = 'hrp'
img_ext = 'png'


###############
## FUNCTIONS ##
###############

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


##############
### SCRIPT ###
##############

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
img = cv2.imread(img_name + '.' + img_ext)
img = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(img.shape)
assert d == 3
image_array = np.reshape(img, (w * h, d))

print("Fitting model on the data")
t0 = time()
image_array_sample = shuffle(image_array)
if (use_color_picking):
    kmeans = KMeans(n_clusters=n_colors, init=colour_clusters).fit(image_array_sample)
else:
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

# count quantity of each label
rgb_vals = kmeans.cluster_centers_* 255
counts = np.zeros(n_colors)
for l in labels:
	counts[l] += 1


# Write count data to csv output
with open (img_name + '_colour_count.csv', 'w') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(['r','g','b','counts'])
	for i in range(n_colors):
		writer.writerow([rgb_vals[i][0], rgb_vals[i][1], rgb_vals[i][2], counts[i]])

# Display quantized and original images
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image')
plt.imshow(img)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.show()
