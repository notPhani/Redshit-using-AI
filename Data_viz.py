# Visualizing the data
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import numpy as np
# Load the image
image_data = fits.getdata('C:/Users/HP/Desktop/Python/data/Galaxy_1/images/frame-r-000756-1-0206.fits')# Replace with your file path

plt.figure(figsize=(8, 8))
plt.imshow(image_data, origin='lower',cmap = 'jet',norm=LogNorm())
plt.title('Masked Image')
plt.colorbar()
plt.show()
