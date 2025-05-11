import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from tensorflow.keras.optimizers import Adam

# Create the autoencoder model for image enhancement
def build_enhancement_model(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    
    # Initial feature extraction
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    
    # First downsampling block
    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Second downsampling block
    x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Multiple residual blocks
    for _ in range(6):
        skip = x
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, skip])
    
    # First upsampling block
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Second upsampling block
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Final reconstruction
    x = Conv2D(3, (3, 3), padding='same')(x)
    outputs = Activation('tanh')(x)
    
    model = Model(inputs, outputs)
    return model

# Preprocess image
def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}. Please check if the file exists and the path is correct.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img

# Create artificial blur
def create_blur(image):
    # Enhanced noise reduction using multiple techniques
    # First apply bilateral filter to preserve edges while reducing noise
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Convert to LAB color space
    # Ensure image is in uint8 format for color conversion
    denoised_uint8 = np.uint8(denoised * 255)
    lab = cv2.cvtColor(denoised_uint8, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge back and convert to RGB
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Convert back to float32 [0,1] range
    enhanced = enhanced.astype('float32') / 255.0
    
    return enhanced

def calculate_metrics(original, enhanced):
    """Calculate PSNR and SSIM metrics"""
    # Convert to uint8 for metric calculation
    orig_uint8 = np.uint8(original * 255)
    enh_uint8 = np.uint8(enhanced * 255)
    
    # Calculate PSNR
    mse = np.mean((orig_uint8 - enh_uint8) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate SSIM
    ssim = structural_similarity(orig_uint8, enh_uint8, channel_axis=2)
    
    return psnr, ssim

def create_enhanced_version(image):
    """Create an enhanced version of the image using traditional CV techniques"""
    # Convert to float32 if not already
    image_f32 = image.astype('float32')
    
    # Apply sharpening using unsharp masking
    blur = cv2.GaussianBlur(image_f32, (0, 0), 3.0)
    sharpened = cv2.addWeighted(image_f32, 1.5, blur, -0.5, 0)
    
    # Enhance contrast using CLAHE
    # Convert to LAB color space
    lab = cv2.cvtColor(np.uint8(sharpened * 255), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Normalize back to [0,1] range
    enhanced = enhanced.astype('float32') / 255.0
    
    # Ensure values are clipped to valid range
    enhanced = np.clip(enhanced, 0, 1)
    
    return enhanced

def custom_loss(y_true, y_pred):
    # Combine MSE with perceptual loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Add total variation loss for spatial smoothness
    tv_loss = tf.reduce_mean(tf.image.total_variation(y_pred))
    
    return mse_loss + 0.1 * tv_loss

# Main execution
def enhance_image(input_image_path, output_path):
    # Load and preprocess image
    original_img = preprocess_image(input_image_path)
    
    # Create enhanced version using traditional methods
    traditionally_enhanced = create_enhanced_version(original_img)
    
    # Calculate metrics for traditional enhancement
    trad_psnr, trad_ssim = calculate_metrics(original_img, traditionally_enhanced)
    
    # Save results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Enhanced\nPSNR: {trad_psnr:.2f}dB\nSSIM: {trad_ssim:.3f}')
    plt.imshow(traditionally_enhanced)
    plt.axis('off')
    
    plt.savefig(output_path)
    plt.close()

# Example usage
if __name__ == "__main__":
    try:
        enhance_image('image.jpg', 'enhanced_results.png')
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")