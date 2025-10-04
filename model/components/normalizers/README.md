# Normalizers

Normalizers rescales activations so they don't explode or vanish. This keeps the values in a predictable range (roughly unit variance):
- preventing extreme activations (stability)
- smooths gradients so backprop doesn't blow up or die out.
- reduces sensitivity to the exact initialization or learning rate.

**It matters in a Transformer because**:
- each token embedding goes through many layers: attention, feedforward, residual adds.
- without normalization, these layers can compound instability - you'd get exploding activations after just a few layers.

**It ensures**:
- Residual connections don't destabilize (residual + large hidden = disaster if not normalized)
- Attention scores stay in reasonable ranges.
- Gradient flow remains stable across 100s of layers and changes does not compounded.
- In short, they ensure scalability in depth and size by making the layers and attentions stable.

**Conceptually**:
- They don't add expressiveness so in theory Architecture can be trained without them.
- They enable scale and convergence (Large models like GPT-3, LLaMa wouldn't train at all otherwise).
- They also act like a implicit regularization:
	- They force (or help) model to focus on patterns rather then magnitude noise, by removing the variance across dimensions.

**In short**:
- Locally (per embedding vector): Normalization just rescales or renormalizes.
- Globally (architecture level):
	- Allows stacking of hundreds of layers
	- Stabilizes training dynamics
	- Enables bigger learning rates and faster convergence
	- Makes models more robust to parameter initialization and scaling.


## 1. RMS Norm
- RMSNorm (Root Mean Square Normalizer) normalizes activations around Root Mean Square (RMS) without centering around the mean, which makes it computationally more efficient compared to LayerNorm. 
- It rescales input vectors by their RMS value and applies a `learnable gain parameter`.
- RMSNorm skips mean substraction ($x - \mu$), reducing computation cost while maintaining training stability (compare to LayerNorm).

### Mathematical Foundation
The normalization formula is:
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{C} \sum_{i=1}^{C} x_i^2 + \epsilon}} \odot g
$$

- $x$: Input vector
- $C$: Number of channels (or dimensions)
- $\epsilon$: Small constant to avoid division by zero
- $g$: Learnable scaling parameter (same dimension as $x$)
- $\odot$: Element-wise multiplication


## 2. LayerNorm
- LayerNorm normalizes activations by centering and scaling across the feature dimension, by subtracting the mean and dividing by the standard deviation, and finally applying learnable affine transformation.

### Mathematical Foundation
The normalization formula is:
$$
\text LayerNorm(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
- $x$: Input vector
- $\mu = \frac{1}{C} \sum_{i=1}^{C}x_i$ is the mean
- $\sigma^2 = \frac{1}{C}\sum_{i=1}^{C}(x_i - \mu)^2$ is the variance
- $\epsilon$ ensures numerical stability
- $\gamma,\beta \in R^C$ are learnable scale and shift parameters 

## Other Normalizers

- **ScaleNorm**: ℓ2 normalization with a scalar gain proposed for improved Transformer training.[](https://arxiv.org/pdf/1910.05895.pdf)
    
- **QK-Norm**: query/key normalization and learnable temperature to reduce softmax saturation.[](https://arxiv.org/pdf/2010.04245.pdf)
    
- **DeepNorm**: residual scaling enabling ultra-deep Transformer stability.[](https://arxiv.org/pdf/2203.00555.pdf)
    
- **Pre-LN vs Post-LN**: analysis of placement and stability trade-offs.[](https://arxiv.org/pdf/2002.04745.pdf)
    
- **CRMSNorm and pre-RMSNorm equivalence**: unifying pre-LN and RMSNorm with compression.[](https://arxiv.org/pdf/2305.14858.pdf)
    
- **FlashNorm**: fused, exact RMSNorm for speed.[](https://arxiv.org/html/2407.09577v1)
    
- **Adoption**: survey-style evidence of RMSNorm prevalence in modern LLM families.