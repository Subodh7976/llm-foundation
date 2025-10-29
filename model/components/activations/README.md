# Activations

Introduces critical non-linearity into Neural Networks.

---
## 1. ReLU (Rectified Linear Unit)

ReLU zeroes negative inputs while passing positive values unchanged, serving as the foundational piecewise linear activation. 
It differs from smooth alternatives by its sharp cutoff at zero and linear positive behavior.

**Mathematical Foundation:**
$$
\text{ReLU(x) = max(0,x)}
$$
where:
- $x$ input tensor of any shape
- Output: same shape as input

**Properties:**
- Derivative: 1 for x>0, 0 for x<0, undefined at x = 0
- Non-saturating for positive inputs
- Zero-centered: No (outputs always non-negative)

#### Optimizations

**Inplace operations:** Setting `inplace=True` reduces memory by 50% during forward pass by overwriting input tensor. Critical for large activations in deep networks. Throughput gain: 5-10% on memory-bound workloads.

**Leaky ReLU dead neuron fix:** Standard ReLU causes ~20-40% of neurons to die (always zero gradient) in deep networks. Leaky ReLU maintains 10-20% gradient flow for negative inputs, improving convergence rate by 1.2-1.5x in practice.

**Fused operations:** Combining bias addition with activation reduces kernel launches from 2 to 1, saving 15-25% latency on small-to-medium tensors where kernel overhead dominates.

---
## 2. GELU (Gaussian Error Linear Unit)

GELU weights inputs by their Gaussian cumulative distribution, providing smooth non-linearity with probabilistic interpretation. 
Unlike ReLU's hard threshold, GELU allows small negative values and exhibits non-monotonic curvature. 

**Mathematical Foundation:**
$$
\text{GELU}(x) = x\cdot \Phi(x) = x \cdot \frac{1}{2}[1+erf(\frac{x}{\sqrt{2}})]
$$
**Variables:**
- $x$ input tensor
- $\Phi(x)$ Standard Gaussian CDF
- $erf(x)$ Error function
$$
erf(x) = \frac{2}{\sqrt{\pi}}\int_0^\pi e^{-t^2}dt
$$

#### Approximations for efficiency:

**Tanh Approximation:**
$$
\text{GELU}_{tanh}(x) = 0.5x
\Big(1+tanh\Big[\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\Big]\Big)
$$
**Sigmoid Approximation:**
$$
\text{GELU}(x) = x\sigma(1.702x)
$$

| Variant      | Speed                                   | Accuracy vs exact       | Description / Notes                              |
| ------------ | --------------------------------------- | ----------------------- | ------------------------------------------------ |
| Exact GELU   | Slow                                    | 100%                    | True Gaussian-based, exact probabilistic meaning |
| Tanh-GELU    | Fast (reduces latency by 50-60%)        | $\approx99.97\%$        | Standard used in almost all LLMs                 |
| Sigmoid-GELU | Very Fast (reduction in latency by 66%) | $\approx98-99\%$        | Simpler logistic approximation                   |
| Fused-GELU   | Fastest                                 | Same as underlying math | Optimized implementation (no math change)        |

**Empirical Gains over ReLU:** GELU consistently improves perplexity by 0.5-1.5 points on language modelling tasks and top-1 accuracy by 0.3-0.8% on ImageNet, attributed to smoother gradients enabling better optimization with Adam/AdamW.

**Memory Efficiency:** All GELU variants have identical memory footprint to ReLU (no learnable parameters, same output size). Choice between variants is purely speed vs. accuracy tradeoff.


---
## Swish / SiLU (Sigmoid Linear Unit)

Swish multiplies input by its sigmoid, creating self-gated smooth activation. It serves as simpler alternative to GELU while maintaining smoothness benefits over ReLU.

**Mathematical Foundation:**
$$
\text{Swish}(x) = x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
$$
**Parameterized variant:**
$$
\text{Swish}_{\beta}(x) = x\cdot\sigma(\beta x)
$$
Variables:
- $x$: input tensor
- $\sigma(x)$ Sigmoid function
- $\beta$ Learnable or fixed scaling parameter (default: 1)

Properties:
- Derivative: $\sigma(x) + x\sigma(x)(1-\sigma(x))$ 
- Self-gating: Output magnitude controlled by input
- Bounded below by $\approx - 0.28$, unbounded above

#### Optimization Techniques

**Fusion benefits:** PyTorch JIT automatically fuses `x * sigmoid(x)` into single kernel, reducing memory reads/writes by 33% and improving throughput by 20-30% on GPU. Essential for large-scale training.

**Learnable beta findings:** Extensive experiments show learned $\beta$ rarely deviates from 1.0 by more than $\pm0.2$, providing <0.1% accuracy gain at cost of additional parameter and gradient computation. Fixed $\beta = 1$ is standard practice. 

**Hard Swish tradeoff:** Piecewise linear approximation achieves 75% latency reduction on mobile CPUs while maintaining 99.5% of Swish accuracy on ImageNet. Critical for edge deployment where FLOPs dominate latency. 

**Empirical performance:** Swish matches or exceeds ReLU by 0.6-1.1% top-1 accuracy on ImageNet across multiple architectures. Improvement attributed to smoother gradient flow and self-gating reducing internal covariate shift. 

---
## SwiGLU (Swish Gated Linear Unit)

SwiGLU combines Swish activation with gated linear units, using separate weight matrices for gating and transformation. This architecture achieves state-of-the-art performance in modern LLMs by balancing expressivity and trainability. 

**Mathematical Foundation:**
$$
\text{SwiGLU}(x, W, V, \beta) = \text{Swish}_\beta(xW)\otimes (xV)
$$
**For FFN Layers:**
$$
FFN_{SwiGLU}(x, W, V, W_2) = (Swish_1(xW)\otimes xV)W_2
$$

**Variables:**
- $x$ Input tensor, shape `(batch, seq_len, d_model)`
- $W, V$ First-layer projection matrices, shape `(d_model, d_ff)`
- $W_2$ Second-layer projection, shape `(d_ff, d_model)`
- $\otimes$ Element-wise multiplication (Hadamard product)
- $\beta$ Typically fixed at 1.0

**Dimensional Scaling:**
- Standard FFN: $d_{ff} = 4 \times d_{model}$ 
- SwiGLU FFN: $d_{ff} = \frac{8}{3} \times d_{model}$ to match parameter count

#### Optimization Explanations

**Fused projection performance:** Concatenating W and V into single weight matrix reduces kernel launches from 2 to 1 for forward pass, improving throughput by 15-25% on A100/H100 GPUs. Memory bandwidth reduced by 33% as input is read once instead of twice.

**Parameter scaling rationale:** Using $d_{ff} = \frac{8}{3} d_{model}$ instead of 4 $d_{model}$ maintains approximately equal parameter count to standard FFN (within 1-2%), enabling fair comparison. The $\frac{8}{3}$ factor accounts for the additional V projection matrix.

**Empirical superiority:** SwiGLU achieves 0.1-0.3 perplexity improvement over GELU and 0.3-0.5 over ReLU on large-scale language modeling at 7B+ parameter scale. Downstream task performance shows 0.5-1.5 point gains on GLUE/SuperGLUE averages. 

**Modern adoptiona rationale:** LLaMA 2/3, Mistral, Qwen 2.5 standardized on SwiGLU because it provides optimal balance of:
- Smooth gradient (better than ReLU)
- Gating expressivity (captures feature interactions)
- Training stability (no divergence issues)
- Computational efficiency (only 10-15% slower than ReLU with fused implementation)

**Memory efficient Note:** SwiGLU requires storing intermediate gate and value activations for backprop, increasing peak memory by ~1.5x compared to standard FFN. Activation checkpointing can reduce this to 1.1x at cost of 20% re-computation overhead. 