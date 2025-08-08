#!/usr/bin/env python3
"""
Validation script to check if the diffusion lab is properly set up
and ready for use with a real HuggingFace token.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment configuration...")
    
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Run setup.sh first.")
        return False
    
    # Check for HuggingFace token
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv("HUGGING_FACE_WRITE_TOKEN")
    if not token or token == "hf_test_token_for_dry_run":
        print("⚠️  No valid HuggingFace token found in .env")
        print("   Get one from: https://huggingface.co/settings/tokens")
        return False
    
    if not token.startswith("hf_"):
        print("⚠️  HuggingFace token format looks incorrect")
        print("   Should start with 'hf_'")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy")
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not found")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: uv pip install -r requirements.txt")
        return False
    
    return True

def check_lab_functionality():
    """Check if the DiffusionLab class works"""
    print("🔍 Checking lab functionality...")
    
    try:
        from diffusion_lab import DiffusionLab
        
        # Test initialization
        lab = DiffusionLab(device="cpu")
        print("✅ DiffusionLab initialization")
        
        # Test basic methods
        info = lab.get_model_info()
        assert info == "No model loaded"
        print("✅ Basic methods working")
        
        return True
        
    except Exception as e:
        print(f"❌ Lab functionality test failed: {e}")
        return False

def check_file_structure():
    """Check if all required files exist"""
    print("🔍 Checking file structure...")
    
    required_files = [
        "src/diffusion_lab.py",
        "notebooks/01_introduction_to_diffusers.ipynb",
        "exercises/exercise_1_basic_generation.py",
        "docs/diffusion_theory.md",
        "docs/troubleshooting.md",
        "requirements.txt",
        "lab_runner.py"
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    return True

def main():
    """Run all validation checks"""
    print("🚀 Validating Diffusion Lab Setup")
    print("=" * 50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Lab Functionality", check_lab_functionality),
        ("Environment", check_environment)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
    
    print(f"\n📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All validation checks passed!")
        print("\n✅ Your diffusion lab is ready to use!")
        print("\nTo start generating images:")
        print("1. python lab_runner.py")
        print("2. jupyter notebook notebooks/")
        print("3. python exercises/exercise_1_basic_generation.py")
        return True
    else:
        print(f"\n❌ {total - passed} validation checks failed.")
        print("\nPlease fix the issues above before using the lab.")
        
        if passed >= 3:
            print("\n💡 Most checks passed - you're almost ready!")
            print("The main issue is likely the HuggingFace token.")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)