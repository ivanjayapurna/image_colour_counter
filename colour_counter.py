import csv
import cv2
import numpy as np

##############
### INPUTS ###
##############

img_name = 'act'
img_ext = 'png'
# proximity ratio (default = 20)
pr = 80


###############
## FUNCTIONS ##
###############

def no_preset_cat():
	unique_colours_r = []
	unique_colours_g = []
	unique_colours_b = []
	counts = []

	for h in range(height):
		print(round(100*h/height,2),'%')
		for w in range(width):
			px_r, px_g, px_b  = img[h,w]
			dne_counter = 0
			for i in range(len(unique_colours_r)):
				if (((px_r <= unique_colours_r[i] + pr) and (px_r >= unique_colours_r[i] - pr)) and ((px_g <= unique_colours_g[i] + pr) and (px_g >= unique_colours_g[i] - pr)) and ((px_b <= unique_colours_b[i] + pr) and (px_b >= unique_colours_b[i] - pr))):
					counts[i] += 1
					break
				else:
					dne_counter += 1
			if dne_counter == len(unique_colours_r):
				unique_colours_r.append(px_r)
				unique_colours_g.append(px_g)
				unique_colours_b.append(px_b)
				counts.append(1)
	print(height,width)
	with open (img_name + '_colour_count.csv', 'a') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(['r','g','b','counts'])
		for i in range(len(unique_colours_r)):
			writer.writerow([unique_colours_r[i], unique_colours_g[i], unique_colours_b[i], counts[i]])


##############
### SCRIPT ###
##############

img = cv2.imread(img_name + '.' + img_ext)
height, width, _ = img.shape
no_preset_cat()
