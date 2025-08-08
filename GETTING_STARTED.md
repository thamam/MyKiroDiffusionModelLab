# 🚀 Getting Started with Diffusion Models Lab

Welcome to your journey into the world of AI image generation! This guide will get you up and running in just a few minutes.

## 🎯 What You'll Build

By the end of this lab, you'll be able to:
- Generate stunning images from text descriptions
- Understand how diffusion models work under the hood
- Experiment with different artistic styles and techniques
- Create your own AI art projects

## ⚡ 5-Minute Quick Start

### Step 1: Get Your HuggingFace Token
1. Go to [HuggingFace](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token (you'll need it in Step 3)

### Step 2: Set Up the Environment
```bash
# Navigate to the lab directory
cd diffusion_lab

# Run the automated setup
chmod +x setup.sh
./setup.sh
```

### Step 3: Add Your Token
```bash
# Edit the .env file
nano .env  # or use your favorite editor

# Add your token:
HUGGING_FACE_WRITE_TOKEN=hf_your_token_here
```

### Step 4: Generate Your First Image!
```bash
# Activate the environment
source .venv/bin/activate

# Run the interactive lab
python lab_runner.py
```

That's it! You should see your first AI-generated image in about 2-3 minutes.

## 🎨 Your First Generation

The lab will start by generating this image:
- **Prompt**: "a beautiful sunset over mountains, digital art"
- **Steps**: 20 (for speed)
- **Style**: Digital art

You'll see the image displayed and saved to the `outputs/` folder.

## 🧭 What's Next?

After your first successful generation, try:

1. **Experiment with prompts**:
   - "a cute robot painting a picture, cartoon style"
   - "a majestic dragon flying over a medieval castle"
   - "a futuristic city with flying cars, cyberpunk style"

2. **Explore the Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/01_introduction_to_diffusers.ipynb
   ```

3. **Try the exercises**:
   ```bash
   python exercises/exercise_1_basic_generation.py
   ```

## 🔧 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space for models

### Recommended Setup
- **GPU**: NVIDIA RTX 3060+ or Apple M1/M2
- **RAM**: 16GB+
- **Storage**: SSD with 20GB+ free space

### Performance Expectations
- **GPU (RTX 3070)**: ~10-15 seconds per image
- **Apple M1/M2**: ~30-45 seconds per image  
- **CPU only**: 2-5 minutes per image

## 🆘 Need Help?

### Common Issues
1. **"No module named 'diffusers'"**
   - Make sure you activated the virtual environment: `source .venv/bin/activate`

2. **"CUDA out of memory"**
   - Try reducing image size or use CPU: `device="cpu"`

3. **"Repository not found"**
   - Check your HuggingFace token is correct and has read permissions

4. **Slow generation**
   - Reduce `num_inference_steps` to 20 for faster results
   - Use a GPU if available

### Get Support
- Check the [troubleshooting guide](docs/troubleshooting.md)
- Read the [theory documentation](docs/diffusion_theory.md)
- Look at example code in the notebooks

## 🎉 Success Tips

1. **Start Simple**: Begin with basic prompts before trying complex scenes
2. **Experiment**: Try different parameters to see their effects
3. **Be Patient**: First model download takes time but only happens once
4. **Save Everything**: Your generated images are saved automatically
5. **Have Fun**: This is creative technology - enjoy the process!

## 🌟 What Makes This Lab Special?

- **Educational Focus**: Learn the theory behind the magic
- **Hands-on Practice**: Generate images from day one
- **Production Ready**: Use the same tools as professionals
- **Comprehensive**: From basics to advanced techniques
- **Well Documented**: Clear explanations and examples

Ready to create some amazing AI art? Let's go! 🎨✨

---

**Next**: Run `python lab_runner.py` and watch the magic happen!