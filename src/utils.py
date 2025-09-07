import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(g_losses, d_losses, d_accs):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(d_accs, label='Discriminator Accuracy')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/training_history.png')
    plt.close()

def generate_interpolation(generator, latent_dim, steps=10):
    """
    Generate interpolations between random points in latent space
    """
    # Generate two random points in latent space
    z1 = np.random.normal(0, 1, (1, latent_dim))
    z2 = np.random.normal(0, 1, (1, latent_dim))
    
    # Interpolate between them
    interpolations = []
    for i in range(steps):
        alpha = i / (steps - 1)
        z = alpha * z1 + (1 - alpha) * z2
        interpolations.append(z)
    
    # Generate images
    interpolations = np.vstack(interpolations)
    generated_images = generator.predict(interpolations, verbose=0)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]
    
    # Plot interpolations
    plt.figure(figsize=(15, 2))
    for i in range(steps):
        plt.subplot(1, steps, i+1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'{i+1}')
    
    plt.tight_layout()
    plt.savefig('output/interpolation.png')
    plt.close()

def save_images(images, epoch, output_dir="output", prefix="letter"):
    """
    Save generated images to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Rescale images from [-1, 1] to [0, 255] if needed
    if images.min() < 0:
        images = (images * 127.5 + 127.5).astype(np.uint8)
    else:
        images = (images * 255).astype(np.uint8)
    
    # Create a grid of images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_{epoch:04d}.png"))
    plt.close()