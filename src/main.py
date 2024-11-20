import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
import numpy as np
from scipy.interpolate import interp1d
from utils import extract_Ra_Dec as Extract
from utils import isolate_galaxy as isolate
from utils import find_pixels as find
from utils import softening_image as smooth
from astropy.io import fits
import matplotlib.pyplot as plt
import xgboost as xgb
import random


model = load_model("C:/Users/HP/Desktop/Python/final_model_2000.keras")

def resample(wave, flux, target_length=4000, epsilon=1e-6):
    # Ensure that the minimum wavelength is greater than zero before applying log10
    wave_min = wave.min()
    wave_max = wave.max()
    
    # If there are non-positive values in wave, adjust them
    if wave_min <= 0:
        wave_min = np.max([wave_max * epsilon, epsilon])  # Make sure it's positive
    if wave_max <= 0:
        wave_max = np.max([wave_min * (1 / epsilon), 1 / epsilon])  # Avoid zero or negative

    # Apply log10 with adjusted minimum and maximum values
    new_waves = np.logspace(np.log10(wave_min), np.log10(wave_max), target_length)
    
    # Interpolate the flux based on the new wavelengths
    interp_func = interp1d(wave, flux, kind="linear", fill_value="extrapolate")
    new_flux = interp_func(new_waves)
    
    return new_waves, new_flux



def stack_input(folder_name):
    spec_folder = os.path.join(folder_name,"spectrum")
    filter_folder = os.path.join(folder_name,"images")

    filters = os.listdir(filter_folder)
    spectrum = os.listdir(spec_folder)[0]

    ra, dec = Extract(os.path.join(spec_folder, spectrum))
    x_pixel, y_pixel = find(os.path.join(filter_folder, filters[0]), ra, dec)

    softened_images = []

    for i in range(len(filters)):
        file_path = os.path.join(filter_folder, filters[i])
        isolated_image = isolate(file_path, x_pixel, y_pixel, show=False)
        softened_image = smooth(isolated_image, show=False)
        softened_images.append(softened_image)
    
    try:
        shapes = [img.shape for img in softened_images]
        if len(set(shapes)) > 1:
            print(f"Your images are in different shapes")
        else:
            stacked_images = np.stack(softened_images, axis=0)  # Fix this line
            return stacked_images
    except Exception as e:
        print(f"Error stacking images: {e}")



def get_spectrum(file_path,show=False):

    with fits.open(file_path,mode="readonly") as file:
        # Assuming the spectrum is in the second HDU
        data = file[1].data["model"]
        waves = 10**file[1].data["loglam"]
        normalized_model = np.log10(data+1e-10)
        #print(normalized_model.shape)# 'flux' column contains the spectrum data
        #flux_length = len(data)
    if show == False:
        return normalized_model,waves
    else:
        plt.plot(waves,normalized_model)
        plt.show()
        return normalized_model,waves



def test_data(galaxy_folder):
    spectrum_folder = f"{galaxy_folder}/spectrum"
    spec_file = os.path.join(spectrum_folder,os.listdir(spectrum_folder)[0])

    x_test = stack_input(galaxy_folder)
    _,data = get_spectrum(spec_file)
    waves,y_test = resample(_,data,4000)
    y_test = y_test/np.max(y_test)
    x_test = np.expand_dims(x_test,axis=-1)

    return x_test,y_test

def predict_spectrum(stacked_images,orginal_spectrum,show = False):
    stacked_input = stacked_images
    #call the model to predict the spectrum
    pred_spectrum = model.predict(stacked_input)

    if show == False:
        return pred_spectrum
    else:
        plt.figure(figsize=(10,5))
        plt.plot(pred_spectrum[0])
        plt.plot(orginal_spectrum,color="red")
        plt.title("Predicted spectrum of the galaxy")
        plt.show()
        return pred_spectrum

def predict_redshift(input_spectrum):
    #load the XGboost model
    loaded_model = xgb.Booster()
    loaded_model.load_model("C:/Users/HP/Desktop/Python/xgboost_model.json")

    test_input = xgb.DMatrix(input_spectrum)

    final_redshift = loaded_model.predict(test_input)

    return final_redshift

if __name__ == "__main__":
    data_folder = "C:/Users/HP/Desktop/Python/Project/data"
    galaxies = os.listdir(data_folder)

    # choose random 5 galaxies and predict their spectrum and their redshift using predicted spectrum

    random_galaxies = random.choices(galaxies,k=6)

    x_test = []
    y_test = []

    for galaxy in random_galaxies:
        x_temp,y_temp = test_data(os.path.join(data_folder,galaxy))
        x_test.append(x_temp)
        y_test.append(y_temp)
    
    pred_spectrums = []

    for i in range(len(x_test)):
        y_pred = predict_spectrum(x_test[i],y_test[i],show=True)
        pred_spectrums.append(y_pred[0])

    pred_redshifts = []

    for spectrum in pred_spectrums:
        red_pred = predict_redshift(spectrum)
        pred_redshifts.append(red_pred)
    print(random_galaxies)
    print(pred_redshifts)
    mean_redshift = []
    for red in pred_redshifts:
        mean_redshift.append(np.mean(red))
    print(mean_redshift)






