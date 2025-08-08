"""
Exercise 1: Basic Image Generation

This exercise will help you practice the fundamentals of text-to-image generation
using the Diffusers library.

Instructions:
1. Complete the functions below
2. Run the script to test your implementations
3. Experiment with different parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diffusion_lab import DiffusionLab
import torch

def exercise_1_simple_generation():
    """
    Exercise 1a: Generate a simple image
    
    Task: Generate an image of "a red apple on a wooden table"
    Requirements:
    - Use 25 inference steps
    - Set guidance scale to 7.5
    - Use seed 123 for reproducibility
    """
    print("🍎 Exercise 1a: Simple Generation")
    
    try:
        # Initialize the DiffusionLab
        lab = DiffusionLab(device="cpu")  # Use CPU for compatibility
        
        # Load the stable diffusion model
        print("Loading model... (this may take a few minutes)")
        lab.load_model("runwayml/stable-diffusion-v1-5")
        
        # Generate the image with the specified parameters
        prompt = "a red apple on a wooden table"
        image = lab.generate_image(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=123
        )
        
        # Display and save the result
        lab.display_image(image, "Exercise 1a: Red Apple")
        lab.save_image(image, "exercise_1a_apple.png")
        print("✅ Exercise 1a completed!")
        
    except Exception as e:
        print(f"❌ Exercise 1a failed: {e}")
        print("Make sure you have a valid HuggingFace token in .env")

def exercise_1_parameter_exploration():
    """
    Exercise 1b: Explore different parameters
    
    Task: Generate the same prompt with different guidance scales
    Requirements:
    - Use prompt: "a majestic lion in the savanna"
    - Try guidance scales: [5.0, 10.0, 15.0]
    - Use 20 inference steps for faster generation
    - Use different seeds for each generation
    """
    print("🦁 Exercise 1b: Parameter Exploration")
    
    lab = DiffusionLab()
    lab.load_model("runwayml/stable-diffusion-v1-5")
    
    prompt = "a majestic lion in the savanna"
    guidance_scales = [5.0, 10.0, 15.0]
    
    # Generate images with different guidance scales
    for i, scale in enumerate(guidance_scales):
        print(f"Generating with guidance scale: {scale}")
        
        try:
            # Generate image with current guidance scale
            image = lab.generate_image(
                prompt=prompt,
                guidance_scale=scale,
                num_inference_steps=20,
                seed=100 + i
            )
            
            lab.save_image(image, f"exercise_1b_lion_scale_{scale}.png")
            
        except Exception as e:
            print(f"❌ Failed to generate with scale {scale}: {e}")
    
    print("✅ Exercise 1b completed! Check your output images.")

def exercise_1_negative_prompts():
    """
    Exercise 1c: Practice with negative prompts
    
    Task: Generate a portrait and improve it with negative prompts
    Requirements:
    - Use prompt: "portrait of a wise old wizard"
    - First generate without negative prompt
    - Then generate with negative prompt: "blurry, low quality, bad anatomy, distorted"
    - Compare the results
    """
    print("🧙 Exercise 1c: Negative Prompts")
    
    lab = DiffusionLab()
    lab.load_model("runwayml/stable-diffusion-v1-5")
    
    prompt = "portrait of a wise old wizard"
    negative_prompt = "blurry, low quality, bad anatomy, distorted"
    
    try:
        # Generate without negative prompt
        print("Generating without negative prompt...")
        image1 = lab.generate_image(
            prompt=prompt,
            num_inference_steps=25,
            seed=42
        )
        
        # Generate with negative prompt
        print("Generating with negative prompt...")
        image2 = lab.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            seed=42
        )
    
        # Save both images for comparison
        lab.save_image(image1, "exercise_1c_wizard_no_negative.png")
        lab.save_image(image2, "exercise_1c_wizard_with_negative.png")
        print("✅ Exercise 1c completed! Compare the two wizard images.")
        
    except Exception as e:
        print(f"❌ Exercise 1c failed: {e}")

def bonus_exercise_batch_generation():
    """
    Bonus Exercise: Batch Generation
    
    Task: Generate multiple variations of the same prompt
    Requirements:
    - Use prompt: "a peaceful zen garden"
    - Generate 4 variations at once
    - Save all variations
    """
    print("🌸 Bonus Exercise: Batch Generation")
    
    lab = DiffusionLab()
    lab.load_model("runwayml/stable-diffusion-v1-5")
    
    prompt = "a peaceful zen garden"
    
    try:
        # Generate 4 variations of the same prompt
        for i in range(4):
            print(f"Generating variation {i+1}/4...")
            image = lab.generate_image(
                prompt=prompt,
                num_inference_steps=20,
                seed=200 + i
            )
            lab.save_image(image, f"bonus_zen_garden_variation_{i+1}.png")
        
        print("✅ Bonus exercise completed!")
        
    except Exception as e:
        print(f"❌ Bonus exercise failed: {e}")

if __name__ == "__main__":
    print("🎯 Starting Diffusion Lab Exercises")
    print("=" * 50)
    
    try:
        exercise_1_simple_generation()
        print()
        
        exercise_1_parameter_exploration()
        print()
        
        exercise_1_negative_prompts()
        print()
        
        bonus_exercise_batch_generation()
        
    except Exception as e:
        print(f"❌ Error during exercises: {e}")
        print("Make sure you have:")
        print("1. Set up your HuggingFace token")
        print("2. Installed all dependencies")
        print("3. Have sufficient GPU memory (or use CPU)")
    
    print("\n🎉 All exercises completed!")
    print("Check the 'outputs' folder for your generated images.")