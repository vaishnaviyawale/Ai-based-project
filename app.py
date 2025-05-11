from flask import Flask, render_template, request, url_for, send_from_directory, redirect, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import uuid
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.optimizers import Adam
import json
from models import db, User

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize database
db.init_app(app)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'static/model'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_enhancement_model(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    
    # Initial feature extraction
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    skip1 = x
    
    # Downsampling path
    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    skip2 = x
    
    x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Residual blocks
    for _ in range(6):
        skip = x
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, skip])
    
    # Upsampling path
    x = UpSampling2D()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Concatenate()([x, skip2])
    
    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Concatenate()([x, skip1])
    
    # Final reconstruction
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    outputs = Activation('tanh')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size while maintaining aspect ratio
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    img = img.astype('float32') / 255.0
    return img

def split_grid_image(image):
    """Split a grid image into individual images"""
    height, width = image.shape[:2]
    cell_height = height // 3
    cell_width = width // 2
    
    cells = []
    for i in range(3):  # 3 rows
        for j in range(2):  # 2 columns
            y = i * cell_height
            x = j * cell_width
            cell = image[y:y + cell_height, x:x + cell_width]
            cells.append(cell)
    return cells

def enhance_image_ml(image_path):
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read the image")
        
        # Convert to YUV color space (Y = luminance, UV = chrominance)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        
        # Apply CLAHE to Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_enhanced = clahe.apply(y)
        
        # Apply bilateral filter for edge preservation and noise reduction
        y_enhanced = cv2.bilateralFilter(y_enhanced, d=5, sigmaColor=10, sigmaSpace=10)
        
        # Apply unsharp mask for sharpening
        gaussian = cv2.GaussianBlur(y_enhanced, (0, 0), 3.0)
        y_enhanced = cv2.addWeighted(y_enhanced, 1.5, gaussian, -0.5, 0)
        
        # Merge back with original chrominance channels
        img_enhanced = cv2.merge([y_enhanced, u, v])
        
        # Convert back to BGR
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_YUV2BGR)
        
        # Apply slight contrast enhancement
        img_enhanced = cv2.convertScaleAbs(img_enhanced, alpha=1.1, beta=5)
        
        return img_enhanced
        
    except Exception as e:
        print(f"ML Enhancement error: {str(e)}")
        # Return the original image in case of error
        return cv2.imread(image_path)

def enhance_image_traditional(image):
    try:
        # Convert to YUV color space (Y = luminance, UV = chrominance)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        
        # Apply CLAHE to Y channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_enhanced = clahe.apply(y)
        
        # Apply bilateral filter for edge preservation and noise reduction
        y_enhanced = cv2.bilateralFilter(y_enhanced, d=5, sigmaColor=10, sigmaSpace=10)
        
        # Merge back with original chrominance channels
        img_enhanced = cv2.merge([y_enhanced, u, v])
        
        # Convert back to BGR
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_YUV2BGR)
        
        # Apply slight contrast enhancement
        img_enhanced = cv2.convertScaleAbs(img_enhanced, alpha=1.1, beta=5)
        
        return img_enhanced
        
    except Exception as e:
        print(f"Traditional Enhancement error: {str(e)}")
        return image

def generate_chart_data(trad_psnr, trad_ssim, ml_psnr, ml_ssim):
    return json.dumps({
        'psnr': {
            'Traditional': float(trad_psnr),
            'ML Enhanced': float(ml_psnr)
        },
        'ssim': {
            'Traditional': float(trad_ssim),
            'ML Enhanced': float(ml_ssim)
        }
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered')

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/reset')
@login_required
def reset():
    """Reset the wizard and return to the initial state"""
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if file and allowed_file(file.filename):
            try:
                # Get enhancement parameters - simplified to just use 'both' by default
                enhancement_type = request.form.get('enhancement_type', 'both')
                noise_type = request.form.get('noise_type', 'gaussian')
                noise_intensity = float(request.form.get('noise_intensity', '0.1'))
                
                unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
                input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                noisy_path = os.path.join(RESULT_FOLDER, f'noisy_{unique_filename}')
                ml_output_path = os.path.join(RESULT_FOLDER, f'ml_enhanced_{unique_filename}')
                trad_output_path = os.path.join(RESULT_FOLDER, f'trad_enhanced_{unique_filename}')
                
                # Save original image
                file.save(input_path)
                
                # Read and process image
                original_img = cv2.imread(input_path)
                if original_img is None:
                    raise ValueError("Could not read the uploaded image")
                
                # Store original dimensions
                original_height, original_width = original_img.shape[:2]
                
                # Convert to RGB and resize for processing
                original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                processed_img = cv2.resize(original_rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                processed_img = processed_img.astype(np.float32) / 255.0
                
                # Create a copy for adding noise (only for display)
                noisy_img = processed_img.copy()
                if noise_type == 'gaussian':
                    noise = np.random.normal(0, noise_intensity, noisy_img.shape)
                    noisy_img = np.clip(noisy_img + noise, 0, 1)
                else:  # salt & pepper
                    noise = np.random.random(noisy_img.shape)
                    noisy_img[noise < noise_intensity/2] = 0
                    noisy_img[noise > 1 - noise_intensity/2] = 1
                
                # Save noisy image (only for display)
                noisy_img_bgr = cv2.cvtColor((noisy_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(noisy_path, noisy_img_bgr)
                
                # Process with traditional method (using original image)
                trad_enhanced = enhance_image_traditional(original_img)
                
                # Process with ML method (using original image)
                ml_enhanced = enhance_image_ml(input_path)
                
                # Resize enhanced images back to original dimensions if they're different
                if (original_height, original_width) != (256, 256):
                    trad_enhanced = cv2.resize(trad_enhanced, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
                    ml_enhanced = cv2.resize(ml_enhanced, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Save enhanced images
                cv2.imwrite(trad_output_path, trad_enhanced)
                cv2.imwrite(ml_output_path, ml_enhanced)
                
                # Calculate metrics (using resized versions for fair comparison)
                processed_original = cv2.resize(original_rgb, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                resized_trad = cv2.resize(cv2.cvtColor(trad_enhanced, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_LANCZOS4)
                resized_ml = cv2.resize(cv2.cvtColor(ml_enhanced, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_LANCZOS4)
                
                trad_psnr = psnr(processed_original, resized_trad)
                ml_psnr = psnr(processed_original, resized_ml)
                
                trad_ssim = ssim(processed_original, resized_trad, multichannel=True, channel_axis=2)
                ml_ssim = ssim(processed_original, resized_ml, multichannel=True, channel_axis=2)
                
                # Get paths for template
                input_display = url_for('static', filename=f'uploads/{unique_filename}')
                noisy_display = url_for('static', filename=f'results/noisy_{unique_filename}')
                ml_output_display = url_for('static', filename=f'results/ml_enhanced_{unique_filename}')
                trad_output_display = url_for('static', filename=f'results/trad_enhanced_{unique_filename}')
                
                chart_data = generate_chart_data(trad_psnr, trad_ssim, ml_psnr, ml_ssim)
                
                return render_template('index.html', 
                                     input_image=input_display,
                                     noisy_image=noisy_display,
                                     ml_output_image=ml_output_display,
                                     trad_output_image=trad_output_display,
                                     trad_psnr=f'{trad_psnr:.2f}',
                                     trad_ssim=f'{trad_ssim:.2f}',
                                     ml_psnr=f'{ml_psnr:.2f}',
                                     ml_ssim=f'{ml_ssim:.2f}',
                                     chart_data=chart_data,
                                     enhancement_type=enhancement_type,
                                     success=True,
                                     data_success=True)
            
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return render_template('index.html', error=str(e))
            
        return render_template('index.html', error='Invalid file type')
    
    return render_template('index.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 