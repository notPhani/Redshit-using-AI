from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage import exposure
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_pixels(file_name,ra,dec):
    try:
        with fits.open(file_name) as hdul:
            # Extract the WCS information
            wcs = WCS(hdul[0].header)
            if not wcs.is_celestial:
                raise ValueError("The object is not celestial.")
            
            # Create a SkyCoord object for the galaxy
            coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')

            # Convert RA, Dec to pixel coordinates
            x_pixel, y_pixel = wcs.world_to_pixel(coord)

            # Debug output
            #print(f"RA, Dec: ({ra}, {dec})")
            #print(f"Pixel coordinates: x={x_pixel}, y={y_pixel}")
            return x_pixel,y_pixel
    except Exception as e:
        print(e)



def isolate_galaxy(file_name, x_pixel, y_pixel, size_pixels_x=110, size_pixels_y=110, show=False):
    """
    Isolate and crop the image around thae given RA and Dec in pixel-based size.

    Parameters:
    - file_name : str
        The file name of the galaxy image.
    - x_pixel : float
        Right Ascension of the galaxy in degrees.
    - y_pixel : float
        Declination of the galaxy in degrees.
    - size_pixels_x : int
        The width of the cropped region in pixels (default is 100 pixels).
    - size_pixels_y : int
        The height of the cropped region in pixels (default is 100 pixels).
    - show : bool
        If True, shows the cropped image. Defaults to False.
    
    Returns:
    - cropped_image : 2D numpy array
        The cropped image data.
    """
    try:
        with fits.open(file_name) as hdul:
            image_data = hdul[0].data
        # Calculate the cropping bounds based on pixel values
        x_min = int(x_pixel - size_pixels_x / 2)
        x_max = int(x_pixel + size_pixels_x / 2)
        y_min = int(y_pixel - size_pixels_y / 2)
        y_max = int(y_pixel + size_pixels_y / 2)

        # Debug output for cropping bounds
        #print(f"Cropping bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

        # Ensure the bounds are within the image dimensions
        x_min, x_max = max(0, x_min), min(image_data.shape[1], x_max)
        y_min, y_max = max(0, y_min), min(image_data.shape[0], y_max)

        # Debug output after adjusting the bounds
        #print(f"Adjusted cropping bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

        # Crop the image
        cropped_image = image_data[y_min:y_max, x_min:x_max]


        # Show the cropped image if 'show' is True
        if show:
            plt.imshow(cropped_image, cmap='jet',norm=LogNorm())
            plt.show()

        return cropped_image

    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
   
#cropped_image = isolate_galaxy("C:/Users/HP/Desktop/Python/data/Galaxy_299489677444933632/images/frame-i-000756-1-0206._2435.fits",146.71421,-1.041304,show=True)

# Extracting the Ra and Dec values from spectrum file for isolation of galaxy

def extract_Ra_Dec(file_name):
    """
    Extracts RA and Dec from the header of an SDSS spectrum FITS file.
    
    Parameters:
    - fits_file : str
        The path to the spectrum FITS file.
    
    Returns:
    - ra : float
        The Right Ascension of the galaxy in degrees.
    - dec : float
        The Declination of the galaxy in degrees.
    """

    try:

        with fits.open(file_name) as file:
            data = file[0].header

            ra = data.get("PLUG_RA")
            dec = data.get("PLUG_DEC")

            if ra is not None and dec is not None:
                return ra, dec
            else:
                raise ValueError(f"Ra and dec not found in the file")
            
    except Exception as e:
        raise ValueError(f"An error has occured {e}")        
    


def softening_image(image, sigma=None, alpha=None, gamma=None, contrast_stretch=True, show=True):
    """
    Creates a high-quality softened galaxy image by blending the original and smoothed images
    with optional contrast enhancement that preserves features.
    
    Parameters:
    - image : 2D numpy array
        Input galaxy image.
    - sigma : float or None
        Standard deviation for Gaussian filter (None will calculate dynamically based on image properties).
    - alpha : float or None
        Blending factor between the original and smoothed image (None will calculate based on image properties).
    - gamma : float or None
        Gamma correction factor (None will calculate dynamically based on image intensity).
    - contrast_stretch : bool
        If True, applies contrast stretching to enhance features.
    - show : bool
        If True, displays the original, smoothed, and blended images.
    
    Returns:
    - blended_image : 2D numpy array
        The high-quality blended galaxy image.
    """
    # Normalize the input image
    image = image/np.max(image)

    # Step 1: Dynamically calculate sigma if not provided
    if sigma is None:
        image_variance = np.var(image)
        if image_variance > 0.02:
            sigma = 2.0
        else:
            sigma = 1.0 

    # Step 2: Dynamically calculate alpha if not provided (based on the image contrast)
    if alpha is None:
        image_std = np.std(image)
        if image_std>0.2:
            alpha = 0.8
        else:
            alpha = 0.6
    
    # Step 3: Dynamically calculate gamma if not provided (based on the image brightness)
    if gamma is None:
        mean_intensity = np.mean(image)
        if mean_intensity<0.4:
            gamma = 1.0
        else:
            gamma = 0.5
    
    # Apply Gaussian smoothing
    smoothed_image = gaussian_filter(image, sigma=sigma)
    
    # Blend the original and smoothed images
    blended_image = alpha * image + (1 - alpha) * smoothed_image
    
    # Optional: Gamma correction for non-linear brightness adjustment
    blended_image = np.float_power(blended_image, gamma)

    if contrast_stretch:
        # Apply logarithmic stretch to enhance contrast without overblowing features
        blended_image = np.log1p(blended_image) 
        blended_image = np.clip(blended_image, 0, 1)  # Normalize back to [0, 1] range
    
    # Visualization
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax1, ax2, ax3 = axes
        
        im1 = ax1.imshow(image, cmap='viridis')
        ax1.set_title("Original Image")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        im2 = ax2.imshow(smoothed_image, cmap='viridis')
        ax2.set_title("Smoothed Image")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        im3 = ax3.imshow(blended_image, cmap='viridis')
        ax3.set_title("Blended Image")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    return blended_image

# Example usage
# Replace 'cropped_image' with your actual galaxy data

#high_quality_blended_image = softening_image(cropped_image, sigma=None, alpha=None, gamma=None, contrast_stretch=True, show=True)
