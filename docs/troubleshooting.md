# Troubleshooting Guide

## Common Issues and Solutions

### 1. HuggingFace Authentication Issues

**Problem**: `401 Unauthorized` or `Repository not found` errors

**Solutions**:
- Ensure your HuggingFace token is set in `.env`
- Verify the token has read permissions
- For gated models, ensure you have access
- Try logging in manually: `huggingface-cli login`

### 2. CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Enable memory efficient attention
pipe.enable_attention_slicing()

# Use lower precision
pipe = pipe.to(torch_dtype=torch.float16)

# Reduce batch size
# Generate one image at a time instead of batches

# Use CPU if necessary
device = "cpu"
```

### 3. Slow Generation on CPU

**Problem**: Very slow image generation

**Solutions**:
- Use GPU if available (CUDA or MPS)
- Reduce number of inference steps
- Use smaller models
- Consider using optimized schedulers like DDIM

### 4. Import Errors

**Problem**: `ModuleNotFoundError` for diffusers or other packages

**Solutions**:
```bash
# Reinstall dependencies
uv pip install -r requirements.txt

# Or install specific packages
uv pip install diffusers transformers torch
```

### 5. Model Loading Issues

**Problem**: Models fail to download or load

**Solutions**:
- Check internet connection
- Clear HuggingFace cache: `rm -rf ~/.cache/huggingface`
- Try a different model
- Use `force_download=True` parameter

### 6. Image Quality Issues

**Problem**: Generated images are blurry or low quality

**Solutions**:
- Increase number of inference steps (50-100)
- Adjust guidance scale (7.5-15.0)
- Use negative prompts
- Try different schedulers
- Improve prompt engineering

### 7. Jupyter Notebook Issues

**Problem**: Notebooks don't start or kernel issues

**Solutions**:
```bash
# Install Jupyter in the virtual environment
uv pip install jupyter ipykernel

# Register the kernel
python -m ipykernel install --user --name diffusion-lab

# Start Jupyter
jupyter notebook
```

### 8. Permission Errors

**Problem**: Permission denied when saving files

**Solutions**:
```bash
# Make sure output directory exists and is writable
mkdir -p outputs
chmod 755 outputs

# Check file permissions
ls -la outputs/
```

## Performance Optimization

### GPU Optimization
```python
# Enable memory efficient attention
pipe.enable_attention_slicing()

# Use xformers if available
pipe.enable_xformers_memory_efficient_attention()

# Use torch.compile for PyTorch 2.0+
pipe.unet = torch.compile(pipe.unet)
```

### Memory Management
```python
# Clear cache between generations
torch.cuda.empty_cache()

# Delete unused pipelines
del pipe
```

## Environment-Specific Issues

### macOS (Apple Silicon)
```python
# Use MPS device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Some operations might need CPU fallback
pipe = pipe.to(device)
```

### Windows
- Ensure you have the correct PyTorch version for your CUDA version
- Use PowerShell or Command Prompt, not Git Bash for some operations
- Path separators should use `\\` or raw strings

### Linux
- Ensure CUDA drivers are properly installed
- Check GPU memory with `nvidia-smi`
- Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

## Getting Help

If you're still having issues:

1. Check the [Diffusers documentation](https://huggingface.co/docs/diffusers)
2. Search [HuggingFace forums](https://discuss.huggingface.co/)
3. Check GitHub issues for the diffusers library
4. Ensure you're using compatible versions of all dependencies

## Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Check System Information
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
```

### Monitor GPU Usage
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```