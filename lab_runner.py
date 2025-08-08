#!/usr/bin/env python3
"""
Diffusion Lab Runner

This script provides an interactive way to run the diffusion models lab.
It guides users through the learning process step by step.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from diffusion_lab import DiffusionLab
from PIL import Image
import matplotlib.pyplot as plt

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")

def main():
    """Main lab runner function"""
    print_header("🌟 Welcome to the Diffusion Models Lab! 🌟")
    
    print("""
This lab will teach you about diffusion models through hands-on examples.
We'll cover:

1. Introduction to Diffusers Library
2. Loading and Using Stable Diffusion
3. Understanding Schedulers
4. Advanced Techniques (Image-to-Image, Inpainting)
5. Custom Model Fine-tuning

Let's get started!
    """)
    
    # Initialize the lab
    print_section("Lab Initialization")
    lab = DiffusionLab()
    
    # Lab 1: Basic Text-to-Image Generation
    print_section("Lab 1: Basic Text-to-Image Generation")
    
    print("Loading Stable Diffusion model...")
    try:
        lab.load_model("runwayml/stable-diffusion-v1-5")
        
        # Generate a simple image
        prompt = "a beautiful sunset over mountains, digital art"
        print(f"Generating image with prompt: '{prompt}'")
        
        image = lab.generate_image(
            prompt=prompt,
            num_inference_steps=20,  # Faster for demo
            seed=42  # For reproducibility
        )
        
        # Display and save the image
        lab.display_image(image, "Lab 1: Basic Generation")
        lab.save_image(image, "lab1_basic_generation.png")
        
    except Exception as e:
        print(f"Error in Lab 1: {e}")
        print("This might be due to missing HuggingFace token or model access issues.")
    
    # Lab 2: Exploring Different Prompts
    print_section("Lab 2: Exploring Different Prompts")
    
    prompts = [
        "a cute robot painting a picture, cartoon style",
        "a majestic dragon flying over a medieval castle",
        "a futuristic city with flying cars, cyberpunk style"
    ]
    
    try:
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/3: '{prompt}'")
            image = lab.generate_image(
                prompt=prompt,
                num_inference_steps=20,
                seed=42 + i
            )
            lab.save_image(image, f"lab2_prompt_{i+1}.png")
            
        print("✅ All images generated! Check the 'outputs' folder.")
        
    except Exception as e:
        print(f"Error in Lab 2: {e}")
    
    # Lab 3: Understanding Guidance Scale
    print_section("Lab 3: Understanding Guidance Scale")
    
    try:
        prompt = "a serene lake with mountains in the background"
        guidance_scales = [1.0, 7.5, 15.0]
        
        images = []
        for scale in guidance_scales:
            print(f"Generating with guidance scale: {scale}")
            image = lab.generate_image(
                prompt=prompt,
                guidance_scale=scale,
                num_inference_steps=20,
                seed=42
            )
            images.append(image)
            lab.save_image(image, f"lab3_guidance_{scale}.png")
        
        # Display comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for img, scale, ax in zip(images, guidance_scales, axes):
            ax.imshow(img)
            ax.set_title(f"Guidance Scale: {scale}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in Lab 3: {e}")
    
    # Lab 4: Model Information
    print_section("Lab 4: Model Information")
    
    info = lab.get_model_info()
    print("Current model information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print_header("🎉 Lab Complete! 🎉")
    print("""
Congratulations! You've completed the basic diffusion models lab.

Next steps:
1. Explore the Jupyter notebooks in the 'notebooks/' folder
2. Try the advanced exercises in the 'exercises/' folder
3. Experiment with different models and parameters

Happy generating! 🎨
    """)

if __name__ == "__main__":
    main()