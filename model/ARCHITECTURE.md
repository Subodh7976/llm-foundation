# TinyStar Architecture Experiments

## Overall Architecture

1. Token Embedding Layer
2. Transformer Block (Loop Layer):
    - **Component Slot 1 (`norm_type`)**: Normalization (`RMSNorm` or `LayerNorm`)
    - **Component Slot 2 (`attention_type`)**: Attention (`MultiHeadAttention`, `GroupedQueryAttention`, `MultiQueryAttention` or `MultiHeadLatentAttention`)
    - **Residual Connection**: Adds input to the attention output.
    - **Component Slot 3 (`norm_type`)**: Normalization (`RMSNorm` or `LayerNorm`)
    - **Component Slot 4 (`ffn_type`)**: Feed-Forward Network (`SwiGLU` or `GELU` based FFN)
    - **Residual Connection**: Adds the output from the first Residual Connection to the FFN output.
3. Normalization Layer
4. Output Head (linear layer that produces Logits)

## Model Configurations

| Model Version | Purpose / Experiment | Attention (`attention_type`) | FFN (`ffn_type`) | Normalization (`norm_type`) | Positional Encoding | Key Innovation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v0.1-Baseline** | A classic GPT-2 style model for baseline performance. | `MHA` (Multi-Head) | `GELU` | `LayerNorm` | `Learned` | The original Transformer architecture. |
| **v0.2-ModernBase** | A modern, strong baseline incorporating SOTA defaults. | `GQA` (Grouped-Query) | `SwiGLU` | `RMSNorm` | `RoPE` | The "Qwen2.5 / Llama" recipe. **Our default.** |
| **v0.3-AttnExp-MHA** | Isolate the effect of MHA vs. the modern baseline. | `MHA` | `SwiGLU` | `RMSNorm` | `RoPE` | Tests if MHA's quality outweighs its inefficiency. |
| **v0.4-AttnExp-MQA** | Isolate the effect of MQA for max inference speed. | `MQA` (Multi-Query) | `SwiGLU` | `RMSNorm` | `RoPE` | Tests max KV cache compression vs. quality loss. |
| **v0.5-AttnExp-MHLA** | Isolate the effect of Multi-Head Latent Attention. | `MHLA` | `SwiGLU` | `RMSNorm` | `RoPE` | Tests DeepSeek's novel KV cache compression. |
| **v0.6-FFNExp-GELU**| Isolate the effect of the FFN, keeping other parts modern. | `GQA` | `GELU` | `RMSNorm` | `RoPE` | Quantifies the benefit of SwiGLU over GELU. |
| **v0.7-LongCtx-SWA**| A model built for long-context efficiency. | `GQA-SWA` (Sliding Window)| `SwiGLU` | `RMSNorm` | `RoPE` | The "Mistral 7B" recipe for efficient long context. |

