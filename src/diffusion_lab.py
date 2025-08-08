"""
Core Diffusion Lab Module

This module provides the main functionality for the diffusion models lab,
including setup, model loading, and image generation utilities.
"""

import os
import torch
from typing import Optional, List, Union
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from huggingface_hub import login
import warnings

# Load environment variables
load_dotenv()

class DiffusionLab:
    """
    Main class for the Diffusion Models Lab
    
    This class provides a structured interface for learning about and
    experimenting with diffusion models.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Diffusion Lab
        
        Args:
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
        """
        self.device = self._setup_device(device)
        self.pipeline = None
        self.current_model = None
        
        # Setup HuggingFace authentication
        self._setup_huggingface_auth()
        
        print(f"🚀 Diffusion Lab initialized on device: {self.device}")
    
    def _setup_device(self, device: Optional[str] = None) -> str:
        """Setup the appropriate device for computation"""
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _setup_huggingface_auth(self):
        """Setup HuggingFace authentication"""
        token = os.getenv("HUGGING_FACE_WRITE_TOKEN")
        if token:
            try:
                login(token=token)
                print("✅ HuggingFace authentication successful")
            except Exception as e:
                print(f"⚠️  HuggingFace authentication failed: {e}")
        else:
            print("⚠️  No HuggingFace token found. Some models may not be accessible.")
    
    def load_model(self, model_id: str, **kwargs):
        """
        Load a diffusion model
        
        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional arguments for pipeline loading
        """
        print(f"📥 Loading model: {model_id}")
        
        try:
            # Load the pipeline
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                **kwargs
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            self.current_model = model_id
            print(f"✅ Model loaded successfully: {model_id}")
            
        except Exception as e:
            print(f"❌ Failed to load model {model_id}: {e}")
            raise
    
    def generate_image(self, 
                      prompt: str, 
                      negative_prompt: Optional[str] = None,
                      num_inference_steps: int = 50,
                      guidance_scale: float = 7.5,
                      width: int = 512,
                      height: int = 512,
                      seed: Optional[int] = None) -> Image.Image:
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: What to avoid in the image
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        if self.pipeline is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        print(f"🎨 Generating image with prompt: '{prompt}'")
        
        # Generate the image
        with torch.autocast(self.device):
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            )
        
        image = result.images[0]
        print("✅ Image generated successfully!")
        
        return image
    
    def display_image(self, image: Image.Image, title: str = "Generated Image"):
        """Display an image with matplotlib"""
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def save_image(self, image: Image.Image, filename: str, output_dir: str = "outputs"):
        """Save an image to disk"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"💾 Image saved to: {filepath}")
        return filepath
    
    def compare_schedulers(self, prompt: str, schedulers: List[str], **kwargs):
        """
        Compare different schedulers with the same prompt
        
        Args:
            prompt: Text prompt to use
            schedulers: List of scheduler names to compare
            **kwargs: Additional generation parameters
        """
        if self.pipeline is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        images = []
        titles = []
        
        original_scheduler = self.pipeline.scheduler
        
        for scheduler_name in schedulers:
            try:
                # Import and set the scheduler
                from diffusers import (DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, 
                                     EulerDiscreteScheduler, DPMSolverMultistepScheduler)
                
                scheduler_map = {
                    'DDIMScheduler': DDIMScheduler,
                    'PNDMScheduler': PNDMScheduler,
                    'LMSDiscreteScheduler': LMSDiscreteScheduler,
                    'EulerDiscreteScheduler': EulerDiscreteScheduler,
                    'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler
                }
                
                if scheduler_name in scheduler_map:
                    scheduler_class = scheduler_map[scheduler_name]
                    self.pipeline.scheduler = scheduler_class.from_config(
                        self.pipeline.scheduler.config
                    )
                    
                    # Generate image
                    image = self.generate_image(prompt, **kwargs)
                    images.append(image)
                    titles.append(f"{scheduler_name}")
                else:
                    print(f"⚠️  Unknown scheduler: {scheduler_name}")
                
            except Exception as e:
                print(f"⚠️  Failed to use scheduler {scheduler_name}: {e}")
        
        # Restore original scheduler
        self.pipeline.scheduler = original_scheduler
        
        # Display comparison
        if images:
            fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
            if len(images) == 1:
                axes = [axes]
            
            for img, title, ax in zip(images, titles, axes):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return images, titles
    
    def get_model_info(self):
        """Get information about the currently loaded model"""
        if self.pipeline is None:
            return "No model loaded"
        
        info = {
            "model_id": self.current_model,
            "device": self.device,
            "scheduler": type(self.pipeline.scheduler).__name__,
            "components": list(self.pipeline.components.keys())
        }
        
        return info