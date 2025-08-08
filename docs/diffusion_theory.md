# Diffusion Models Theory

## Introduction

Diffusion models are a class of generative models that have revolutionized image generation. They work by learning to reverse a gradual noising process, allowing them to generate high-quality images from pure noise.

## The Diffusion Process

### Forward Process (Noising)

The forward process gradually adds Gaussian noise to an image over T timesteps:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Where:
- `x_0` is the original image
- `x_T` is pure noise
- `β_t` is the noise schedule

### Reverse Process (Denoising)

The reverse process learns to remove noise step by step:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

The neural network learns to predict the noise that was added at each step.

## Key Components

### 1. U-Net Architecture

The core of most diffusion models is a U-Net architecture that:
- Takes a noisy image and timestep as input
- Predicts the noise to be removed
- Uses skip connections for better gradient flow
- Incorporates attention mechanisms for global context

### 2. Noise Schedulers

Different schedulers control how noise is added/removed:

- **DDPM (Denoising Diffusion Probabilistic Models)**: Original scheduler
- **DDIM (Denoising Diffusion Implicit Models)**: Faster, deterministic sampling
- **LMS (Linear Multi-Step)**: Uses linear multi-step methods
- **Euler**: Based on Euler's method for ODEs

### 3. Conditioning

For text-to-image generation, models use:
- **Text Encoder**: Converts text to embeddings (usually CLIP)
- **Cross-Attention**: Injects text information into the U-Net
- **Classifier-Free Guidance**: Improves prompt following

## Mathematical Foundation

### Variational Lower Bound

The training objective is to maximize the likelihood:

```
L = E_q[log p_θ(x_0 | x_1)] - D_KL(q(x_T | x_0) || p(x_T)) - Σ_{t=2}^T E_q[D_KL(q(x_{t-1} | x_t, x_0) || p_θ(x_{t-1} | x_t))]
```

### Simplified Training Objective

In practice, the loss is simplified to:

```
L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t, t)||²]
```

Where the model learns to predict the noise `ε` that was added.

## Stable Diffusion Architecture

Stable Diffusion operates in latent space for efficiency:

1. **VAE Encoder**: Compresses images to latent space
2. **U-Net**: Performs diffusion in latent space
3. **VAE Decoder**: Converts back to pixel space
4. **Text Encoder**: CLIP text encoder for conditioning

## Guidance Techniques

### Classifier Guidance

Uses a separate classifier to guide generation:
```
ε̃_θ(x_t, t) = ε_θ(x_t, t) - s∇_{x_t} log p_φ(y | x_t)
```

### Classifier-Free Guidance

Trains the model both with and without conditioning:
```
ε̃_θ(x_t, t) = ε_θ(x_t, t, ∅) + s(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```

Where `s` is the guidance scale.

## Advantages of Diffusion Models

1. **High Quality**: Produce very high-quality images
2. **Stable Training**: More stable than GANs
3. **Flexible**: Can be conditioned on various inputs
4. **Controllable**: Various techniques for controlling generation

## Limitations

1. **Slow Sampling**: Requires many denoising steps
2. **Computational Cost**: Expensive to train and run
3. **Memory Requirements**: Large models need significant GPU memory

## Recent Advances

- **Latent Diffusion**: Operating in compressed latent space
- **Consistency Models**: Single-step generation
- **ControlNet**: Additional spatial control
- **LoRA**: Efficient fine-tuning techniques