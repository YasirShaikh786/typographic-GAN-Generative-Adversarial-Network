import numpy as np
import os
import argparse
from data_loader import create_letter_dataset
from gan_model import SimpleGlyphGAN

def main():
    parser = argparse.ArgumentParser(description='Train a GAN to generate typographic characters')
    parser.add_argument('--letter', type=str, default='A', help='Letter to generate')
    parser.add_argument('--img_size', type=int, default=28, help='Image size (width and height)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--sample_interval', type=int, default=100, help='Interval for sampling images')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Create dataset
    print(f"Creating dataset for letter '{args.letter}'...")
    X_train = create_letter_dataset(
        letter=args.letter, 
        img_size=args.img_size, 
        num_samples=args.num_samples
    )
    
    print(f"Dataset shape: {X_train.shape}")
    
    # Create and train GAN
    gan = SimpleGlyphGAN(
        img_rows=args.img_size, 
        img_cols=args.img_size
    )
    
    print("Starting training...")
    gan.train(
        X_train, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        sample_interval=args.sample_interval
    )

if __name__ == '__main__':
    main()