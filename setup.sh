#!/bin/bash

# Diffusion Lab Setup Script
# This script sets up the complete environment for the diffusion models lab

echo "🚀 Setting up Diffusion Models Lab..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Then restart your terminal and run this script again."
    exit 1
fi

echo "✅ UV found"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch first (CPU version for compatibility)
echo "📥 Installing PyTorch (CPU version)..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install core diffusion dependencies
echo "📥 Installing diffusion libraries..."
uv pip install diffusers transformers accelerate

# Install remaining dependencies
echo "📥 Installing remaining dependencies..."
uv pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p outputs
mkdir -p cache

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your HuggingFace token!"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
chmod +x lab_runner.py
find exercises -name "*.py" -exec chmod +x {} \;

# Run setup tests
echo "🧪 Running setup tests..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete and tested!"
    echo ""
    echo "Next steps:"
    echo "1. Get a HuggingFace token: https://huggingface.co/settings/tokens"
    echo "2. Edit .env and add your HUGGING_FACE_WRITE_TOKEN"
    echo "3. Activate the virtual environment: source .venv/bin/activate"
    echo "4. Run the lab: python lab_runner.py"
    echo "5. Or start Jupyter: jupyter notebook notebooks/"
    echo ""
    echo "For GPU support, reinstall PyTorch with CUDA:"
    echo "uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "Happy learning! 🎨✨"
else
    echo ""
    echo "❌ Setup tests failed. Please check the errors above."
    exit 1
fi