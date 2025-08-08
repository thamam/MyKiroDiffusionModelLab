#!/usr/bin/env python3
"""
Test version of lab runner that doesn't download models
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_lab_runner_imports():
    """Test that lab runner can import everything it needs"""
    print("🧪 Testing lab runner imports...")
    
    try:
        from diffusion_lab import DiffusionLab
        from PIL import Image
        import matplotlib.pyplot as plt
        print("✅ All lab runner imports successful")
        return True
    except Exception as e:
        print(f"❌ Lab runner import failed: {e}")
        return False

def test_lab_initialization():
    """Test lab initialization without model loading"""
    print("🧪 Testing lab initialization...")
    
    try:
        from diffusion_lab import DiffusionLab
        
        # Test with CPU device
        lab = DiffusionLab(device="cpu")
        
        # Test basic methods
        info = lab.get_model_info()
        assert info == "No model loaded"
        
        print("✅ Lab initialization successful")
        return True
    except Exception as e:
        print(f"❌ Lab initialization failed: {e}")
        return False

def test_directory_creation():
    """Test that output directories can be created"""
    print("🧪 Testing directory creation...")
    
    try:
        import os
        
        # Test creating outputs directory
        os.makedirs("outputs", exist_ok=True)
        assert os.path.exists("outputs")
        
        print("✅ Directory creation successful")
        return True
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        return False

def test_exercise_file_syntax():
    """Test that exercise files have valid Python syntax"""
    print("🧪 Testing exercise file syntax...")
    
    try:
        import ast
        
        exercise_file = Path("exercises/exercise_1_basic_generation.py")
        if exercise_file.exists():
            with open(exercise_file, 'r') as f:
                content = f.read()
            
            # Parse the Python code to check syntax
            ast.parse(content)
            print("✅ Exercise file syntax is valid")
            return True
        else:
            print("❌ Exercise file not found")
            return False
            
    except SyntaxError as e:
        print(f"❌ Exercise file syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Exercise file test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Lab Runner Tests (No Model Download)")
    print("=" * 60)
    
    tests = [
        test_lab_runner_imports,
        test_lab_initialization,
        test_directory_creation,
        test_exercise_file_syntax
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All lab runner tests passed!")
        print("\n✅ The lab is ready to use with a real HuggingFace token")
        print("\nTo test with actual model generation:")
        print("1. Get a HuggingFace token from https://huggingface.co/settings/tokens")
        print("2. Update .env with your real token")
        print("3. Run: python lab_runner.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)