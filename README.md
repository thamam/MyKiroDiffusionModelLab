# 🧨 Diffusion Models Lab - HuggingFace Course Recreation

This comprehensive lab recreates and expands upon the HuggingFace diffusion models course, providing a structured, educational environment for learning about state-of-the-art image generation using diffusion models.

## 🎯 What You'll Learn

- **Diffusion Model Fundamentals**: Understanding the theory behind diffusion models
- **HuggingFace Diffusers Library**: Hands-on experience with the industry-standard library
- **Text-to-Image Generation**: Creating images from text descriptions
- **Advanced Techniques**: Schedulers, guidance scales, negative prompts, and more
- **Model Comparison**: Exploring different models and their capabilities
- **Best Practices**: Optimization, troubleshooting, and production considerations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- UV package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- HuggingFace account with write token ([get yours here](https://huggingface.co/settings/tokens))
- GPU recommended (CUDA or Apple Silicon), but CPU works too

### Automated Setup

```bash
cd diffusion_lab
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Set up the environment**:
   ```bash
   cd diffusion_lab
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your HUGGING_FACE_WRITE_TOKEN
   ```

4. **Start learning**:
   ```bash
   # Interactive Python lab
   python lab_runner.py
   
   # Or use Jupyter notebooks
   jupyter notebook notebooks/
   ```

## 📚 Lab Structure

```
diffusion_lab/
├── 📓 notebooks/           # Interactive Jupyter notebooks
│   └── 01_introduction_to_diffusers.ipynb
├── 🧪 exercises/          # Hands-on coding exercises
│   └── exercise_1_basic_generation.py
├── 📦 src/                # Core lab modules
│   └── diffusion_lab.py   # Main DiffusionLab class
├── 📖 docs/               # Documentation and theory
│   ├── diffusion_theory.md
│   └── troubleshooting.md
├── 🖼️ outputs/            # Generated images (created automatically)
├── 🔧 cache/              # Model cache (created automatically)
└── 📋 requirements.txt    # Dependencies
```

## 🎓 Learning Path

### 1. **Theory Foundation** (15 mins)
   - Read `docs/diffusion_theory.md`
   - Understand the mathematical foundations

### 2. **Interactive Lab** (30 mins)
   - Run `python lab_runner.py`
   - Follow the guided exercises

### 3. **Jupyter Exploration** (45 mins)
   - Open `notebooks/01_introduction_to_diffusers.ipynb`
   - Work through all examples interactively

### 4. **Hands-on Exercises** (30 mins)
   - Complete `exercises/exercise_1_basic_generation.py`
   - Experiment with your own prompts

### 5. **Advanced Exploration** (Open-ended)
   - Try different models and techniques
   - Create your own artistic projects

## 🛠️ Key Features

### DiffusionLab Class
The core `DiffusionLab` class provides a clean, educational interface:

```python
from src.diffusion_lab import DiffusionLab

# Initialize the lab
lab = DiffusionLab(device="auto")

# Load a model
lab.load_model("runwayml/stable-diffusion-v1-5")

# Generate an image
image = lab.generate_image(
    prompt="a beautiful sunset over mountains",
    num_inference_steps=50,
    guidance_scale=7.5
)

# Display and save
lab.display_image(image)
lab.save_image(image, "sunset.png")
```

### Supported Models
- Stable Diffusion v1.5
- Stable Diffusion v2.1
- Any HuggingFace diffusion model
- Custom fine-tuned models

### Advanced Features
- Multiple scheduler comparison
- Batch generation
- Negative prompting
- Parameter exploration tools
- Memory optimization
- Cross-platform support (CUDA, MPS, CPU)

## 🎨 Example Outputs

The lab will generate various images demonstrating:
- Basic text-to-image generation
- Different art styles (oil painting, watercolor, digital art)
- Parameter effects (guidance scale, inference steps)
- Scheduler comparisons
- Negative prompt improvements

## 🔧 Troubleshooting

Having issues? Check our comprehensive [troubleshooting guide](docs/troubleshooting.md) covering:
- HuggingFace authentication
- CUDA memory issues
- Performance optimization
- Platform-specific problems

## 📖 Additional Resources

- [HuggingFace Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Original HuggingFace Course](https://github.com/huggingface/diffusion-models-class)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)

## 🤝 Contributing

This is an educational project! Feel free to:
- Add new exercises
- Improve documentation
- Create additional notebooks
- Share your generated artwork

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying models and libraries.

---

**Ready to start creating amazing AI art? Let's dive in!** 🎨✨