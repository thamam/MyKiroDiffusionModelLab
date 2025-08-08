#!/usr/bin/env python3
"""
Test script to verify the diffusion lab setup works correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        from PIL import Image
        print("✅ PIL/Pillow")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
        
        from dotenv import load_dotenv
        print("✅ python-dotenv")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_diffusion_lab_class():
    """Test the DiffusionLab class initialization"""
    print("\n🧪 Testing DiffusionLab class...")
    
    try:
        from diffusion_lab import DiffusionLab
        
        # Test initialization
        lab = DiffusionLab(device="cpu")  # Use CPU for testing
        print("✅ DiffusionLab initialized successfully")
        
        # Test device setup
        assert lab.device == "cpu"
        print("✅ Device setup working")
        
        # Test model info without loading a model
        info = lab.get_model_info()
        assert info == "No model loaded"
        print("✅ Model info method working")
        
        return True
        
    except Exception as e:
        print(f"❌ DiffusionLab test failed: {e}")
        return False

def test_environment_setup():
    """Test environment file setup"""
    print("\n🧪 Testing environment setup...")
    
    try:
        # Check if .env.example exists
        env_example = Path(".env.example")
        if env_example.exists():
            print("✅ .env.example found")
        else:
            print("❌ .env.example not found")
            return False
        
        # Check if directories exist
        dirs_to_check = ["src", "notebooks", "exercises", "docs"]
        for dir_name in dirs_to_check:
            if Path(dir_name).exists():
                print(f"✅ {dir_name}/ directory exists")
            else:
                print(f"❌ {dir_name}/ directory missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_notebook_syntax():
    """Test that the notebook file is valid JSON"""
    print("\n🧪 Testing notebook syntax...")
    
    try:
        import json
        
        notebook_path = Path("notebooks/01_introduction_to_diffusers.ipynb")
        if notebook_path.exists():
            with open(notebook_path, 'r') as f:
                json.load(f)  # This will raise an exception if invalid JSON
            print("✅ Notebook JSON syntax is valid")
            return True
        else:
            print("❌ Notebook file not found")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ Notebook JSON syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Notebook test failed: {e}")
        return False

def test_requirements_file():
    """Test that requirements.txt is valid"""
    print("\n🧪 Testing requirements.txt...")
    
    try:
        req_path = Path("requirements.txt")
        if req_path.exists():
            with open(req_path, 'r') as f:
                lines = f.readlines()
            
            # Check for key packages
            content = ''.join(lines).lower()
            required_packages = ['diffusers', 'torch', 'transformers', 'pillow', 'matplotlib']
            
            for package in required_packages:
                if package in content:
                    print(f"✅ {package} found in requirements")
                else:
                    print(f"❌ {package} missing from requirements")
                    return False
            
            return True
        else:
            print("❌ requirements.txt not found")
            return False
            
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Diffusion Lab Setup Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_diffusion_lab_class,
        test_environment_setup,
        test_notebook_syntax,
        test_requirements_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The lab setup is working correctly.")
        print("\nNext steps:")
        print("1. Add your HuggingFace token to .env")
        print("2. Run: python lab_runner.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)