import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filelist = glob.glob('NonCarDataset/*')
for f in filelist:
	x = np.array(Image.open(f))
	if(x.shape == (32,32)):
		imgplot = plt.imshow(x)
		plt.show()
