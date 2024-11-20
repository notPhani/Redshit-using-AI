import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Input, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from keras.utils import Progbar
import random
from tensorflow.image import resize
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import HeNormal, GlorotUniform
import tensorflow as tf
from scipy.interpolate import interp1d
from utils import isolate_galaxy as isolate
from utils import find_pixels as find
from utils import softening_image as smooth
from utils import extract_Ra_Dec as Extract
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


# List all physical devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs: ", physical_devices)

# If you have GPUs, set memory growth
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU devices found.")
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

    ra,dec = Extract(os.path.join(spec_folder,spectrum))
    x_pixel,y_pixel = find(os.path.join(filter_folder,filters[0]),ra,dec)

    softened_images = []

    for i in range(len(filters)):
        file_path = os.path.join(filter_folder,filters[i])
        isolated_image = isolate(file_path,x_pixel,y_pixel,show=False)
        softened_image = smooth(isolated_image,show=False)
        softened_images.append(softened_image)
    
    try:
        shapes = [img.shape for img in softened_images]
        if len(set(shapes)) >1:
            print(f"Your images are in different shapes")
        else:
            stacked_input = np.stack(softened_image,axis=0)
            return stacked_input
    except Exception as e:
        print(f"{e}")


# print(stack_input('C:/Users/HP/Desktop/Python/Project/data/Galaxy_299561695456552960'))

# Length of spectrum is 3806

def get_spectrum(file_path,show=False):

    with fits.open(file_path) as file:
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

from keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import layers,models
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def create_cnn(input_shape,output_shape):
    model = models.Sequential()

    # First Convolution Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # Adjust input shape based on your data
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolution Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the 2D output to 1D
    model.add(layers.Flatten())

    # Dense layer to output the spectrum (assuming 4000 flux values)
    model.add(layers.Dense(4000, activation='linear'))  # Linear activation for regression tasks

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(galaxy_folder, epochs, batch_size, max_galaxies):
    max_galaxies = len(galaxy_folder)
    x_train, y_train = [], []
    wavelength = list(set([]))
    galaxies = os.listdir(galaxy_folder)[:max_galaxies]

    # Prepare the data for training
    for galaxy in galaxies:
        spec_folder = os.path.join(galaxy_folder, galaxy, "spectrum")
        filter_folder = os.path.join(galaxy_folder, galaxy)

        stacked_input = stack_input(filter_folder)

        spec_file = os.path.join(spec_folder, os.listdir(spec_folder)[0])

        waves, data = get_spectrum(spec_file)
        waves, data = resample(waves, data, target_length=4000)

        wavelength.append(waves)
        x_train.append(stacked_input)
        y_train.append(data)

    # Normalize the data and convert into numpy array
    y_train = np.array(y_train) / np.max(y_train)
    x_train = np.expand_dims(np.array(x_train), axis=-1)

    model = create_cnn(x_train.shape[1:], 4000)
    progbar = Progbar(target=epochs)

    # Start the training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train for one epoch
        history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1)
        progbar.update(epoch + 1)

        # Check every 10th epoch
        if (epoch + 1) % 2000 == 0:
            # Print or save results for monitoring
            print(f"\nResults at Epoch {epoch + 1}:")
            print(f"Loss: {history.history['loss'][0]:.4f}, MAE: {history.history['mae'][0]:.4f}")
            
            # Optionally, plot predictions on a random test sample
            # For this example, use the first galaxy in the training data (or any test set)
            test_idx = random.randint(0,max_galaxies-1)
            pred_spectrum = model.predict(np.expand_dims(x_train[test_idx], axis=0))
            
            # Plot the true vs predicted spectrum
            plt.figure(figsize=(10, 5))
            print(pred_spectrum.shape)
            plt.plot(y_train[test_idx], label='True Spectrum', color='blue')
            plt.plot(pred_spectrum[0], label='Predicted Spectrum', color='black')
            plt.title(f"True vs Predicted Spectrum at Epoch {epoch + 1}")
            plt.xlabel("Wavelength (Index)")
            plt.ylabel("Flux")
            plt.legend()
            plt.show()
        
    model.save("C:/Users/HP/Desktop/Python/Project/final_model.keras")
    return model


# Call the function with your dataset
model = train_model('C:/Users/HP/Desktop/Python/Project/data', epochs=100, batch_size=512,max_galaxies=None)


