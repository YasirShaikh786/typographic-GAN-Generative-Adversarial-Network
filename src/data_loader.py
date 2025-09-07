import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_letter_dataset(letter='A', img_size=28, num_samples=1000, noise_level=25):
    """
    Create a dataset of a specific letter with variations
    """
    images = []
    
    # Try to use a font, fallback to drawing if not available
    try:
        font = ImageFont.truetype("arial.ttf", img_size - 4)
    except:
        font = ImageFont.load_default()
    
    for _ in range(num_samples):
        # Create a blank image
        img = Image.new('L', (img_size, img_size), color=0)  # 'L' for grayscale
        draw = ImageDraw.Draw(img)
        
        # Draw the letter
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (img_size - text_width) / 2 - bbox[0]
        y = (img_size - text_height) / 2 - bbox[1]
        
        draw.text((x, y), letter, fill=255, font=font)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Add some noise to create variations
        noise = np.random.normal(0, noise_level, (img_size, img_size))
        img_array = np.clip(img_array + noise, 0, 255)
        
        images.append(img_array)
    
    images = np.array(images)
    images = images.reshape(images.shape[0], img_size, img_size, 1)
    return images

def load_real_font_dataset(font_path, img_size=28, chars=None):
    """
    Load real font characters if you have font files
    """
    if chars is None:
        chars = ['A', 'B', 'C', 'D', 'E']  # Default characters
    
    images = []
    labels = []
    
    try:
        font = ImageFont.truetype(font_path, img_size - 4)
    except:
        print(f"Font {font_path} not found, using default font")
        font = ImageFont.load_default()
    
    for char in chars:
        # Create a blank image
        img = Image.new('L', (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)
        
        # Draw the character
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (img_size - text_width) / 2 - bbox[0]
        y = (img_size - text_height) / 2 - bbox[1]
        
        draw.text((x, y), char, fill=255, font=font)
        
        # Convert to numpy array
        img_array = np.array(img)
        img_array = img_array.reshape(img_size, img_size, 1)
        
        images.append(img_array)
        labels.append(char)
    
    return np.array(images), labels

def save_images(images, epoch, output_dir="output", prefix="letter"):
    """
    Save generated images to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Rescale images from [-1, 1] to [0, 255]
    images = (images * 127.5 + 127.5).astype(np.uint8)
    
    # Create a grid of images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{epoch:04d}.png"))
    plt.close()