from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def isolate_galaxy(file_name, ra, dec, size_pixels_x=110, size_pixels_y=110, show=False):
    """
    Isolate and crop the image around the given RA and Dec in pixel-based size.

    Parameters:
    - file_name : str
        The file name of the galaxy image.
    - ra : float
        Right Ascension of the galaxy in degrees.
    - dec : float
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
            # Extract the WCS information
            wcs = WCS(hdul[0].header)
            image_data = hdul[0].data  # Image data

            if not wcs.is_celestial:
                raise ValueError("The object is not celestial.")
            
            # Create a SkyCoord object for the galaxy
            coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')

            # Convert RA, Dec to pixel coordinates
            x_pixel, y_pixel = wcs.world_to_pixel(coord)

            # Debug output
            print(f"RA, Dec: ({ra}, {dec})")
            print(f"Pixel coordinates: x={x_pixel}, y={y_pixel}")

            # Calculate the cropping bounds based on pixel values
            x_min = int(x_pixel - size_pixels_x / 2)
            x_max = int(x_pixel + size_pixels_x / 2)
            y_min = int(y_pixel - size_pixels_y / 2)
            y_max = int(y_pixel + size_pixels_y / 2)

            # Debug output for cropping bounds
            print(f"Cropping bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

            # Ensure the bounds are within the image dimensions
            x_min, x_max = max(0, x_min), min(image_data.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(image_data.shape[0], y_max)

            # Debug output after adjusting the bounds
            print(f"Adjusted cropping bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

            # Crop the image
            cropped_image = image_data[y_min:y_max, x_min:x_max]

            # Show the cropped image if 'show' is True
            if show:
                plt.imshow(cropped_image, cmap='jet', norm=LogNorm())
                plt.title(f"Cropped image around RA={ra}, Dec={dec}")
                plt.show()

            return cropped_image

    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

# Example usage
file_name = 'C:/Users/HP/Desktop/Python/data/Galaxy_299489677444933632/images/frame-r-000756-1-0206._2437.fits'
ra = 146.71421  # Example RA
dec =-1.041304  # Example Dec

cropped_image = isolate_galaxy(file_name, ra, dec, size_pixels_x=100, size_pixels_y=100, show=True)
print(f"Cropped image shape: {cropped_image.shape}")
