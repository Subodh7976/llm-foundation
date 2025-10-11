# Embeddings

- Used for converting basic token (integer) based representation into Vector representation for pattern identification. 
- It basically maps a discrete token to a continuous vector in $R^C$

Why is this important:
- Turns symbolic language into something neural networks can operate on. (vectors)
- Each dimension captures some latent property (semantics, syntax, morphology etc. - though not interpretable per se)
- Language -> vector bridge
- Good embeddings -> easier pattern discovery -> faster convergence
- Trainable -> optimized jointly with rest of the model and performs preliminary task of semantic understanding or representation of the sequence or token.
- Connects discrete language to the continuous reasoning space.

## 1. Token Embedding

- Converts token IDs into continuous dense vector representations through a learnable lookup table.
- Each token in the vocabulary maps to a unique embedding vector that captures semantic information learned during training.

### Mathematical Foundation
$$
\text Embed(token\_id) = W[token\_id]
$$
where:
- $W \in R^{VXC}$ is the embedding weight matrix
- $V$ is the vocabulary size
- $C$ is the embedding dimension
- The operation is a simple row lookup/indexing.

## 2. Learned Positional Embedding

- Provides absolute position information by maintaining a trainable lookup table that maps each position index to a unique dense vector.
- These embeddings are added element-wise to `Token Embeddings`, enabling the attention mechanism to distinguish token order.

### Mathematical Foundation
$$
\text PosEmbed(position) = P[position]
$$
where:
- $P \in R^{T_{max} X C}$ is the position embedding matrix
- $T_{max}$ is the maximum context length.
- $C$ is the embedding dimension
- Position index ranges from 0 to $T_{max} - 1$

Final Embedding representation which combines token and positional embedding:
$$
\text Output = TokenEmbed(x) + PosEmbed(positions)
$$

## 3. Sinusoidal Positional Embedding

- Provides fixed, deterministic position information using sine and cosine functions of different frequencies. 
- Require no parameters and are computed using mathematical functions, resulting in strong inductive bias for modeling relative positions.

### Mathematical Foundation
$$
PE_{(pos, 2i)} = sin(\frac{pos}{10000^{2i/d_{model}}})
$$
$$
PE_{(pos, 2i+1)} = cos(\frac{pos}{10000^{2i/d_{model}}})
$$
where:
- $pos$ is the position index (0 to $T$ - 1)
- $i$ is the dimension index (0 to $d_{model}/2$ - 1)
- Even dimensions use sine, odd dimensions use cosine
- Even dimensions has a different frequency, creating a unique encoding per position

## 3. Rotary Positional Embedding (RoPE)

- Encodes positional information by applying position-dependent rotations to query and key vectors in 2D subspaces.
- Unlike absolute positional encodings added to the embeddings, RoPE directly modifies attention scores to encode relative positions through rotational transformations, providing superior extrapolation to longer contexts.

### Mathematical Foundation
- Rope applies a rotation matrix to each consecutive pair of features in Q and K vectors:

$$
f(x, m) =
\begin{bmatrix}
x_0 \\
x_1 \\
x_2 \\
x_3 \\
\vdots \\
x_{d-2} \\
x_{d-1}
\end{bmatrix}
\otimes
\begin{bmatrix}
\cos(m\theta_0) \\
\cos(m\theta_0) \\
\cos(m\theta_1) \\
\cos(m\theta_1) \\
\vdots \\
\cos\left(m\theta_{\frac{d}{2}-1}\right) \\
\cos\left(m\theta_{\frac{d}{2}-1}\right)
\end{bmatrix}
+
\begin{bmatrix}
-x_1 \\
x_0 \\
-x_3 \\
x_2 \\
\vdots \\
-x_{d-1} \\
x_{d-2}
\end{bmatrix}
\otimes
\begin{bmatrix}
\sin(m\theta_0) \\
\sin(m\theta_0) \\
\sin(m\theta_1) \\
\sin(m\theta_1) \\
\vdots \\
\sin\left(m\theta_{\frac{d}{2}-1}\right) \\
\sin\left(m\theta_{\frac{d}{2}-1}\right)
\end{bmatrix}
$$

where:
- $m$ is the position index
- $\theta_i = 10000^{-2i/d}$ is the frequency for dimension pair $i$
- $d$ is the head dimension (must be even)
- $\otimes$ denotes element-wise multiplication
For queries at position m and keys at position n, the dot product naturally encodes relative position $m - n$ through the rotation difference, making attention inherently position-aware.
After rotation, $q_m^Tk_n = q^TR_m^TR_nk = q^TR_{m-n}k$ where $R_{\theta}$ is the rotation matrix.

## 4. ALiBi

- ALiBi (Attention with Linear Biases) directly adds linearly decreasing penalties to attention scores based on query-key distance, eliminating positional embeddings entirely while enabling length extrapolation. 
- Unlike sinusoidal embedding that embed positions as vectors added to tokens, ALiBi modifies attention computation itself by biasing scores with simple negative slopes proportional to token separation.

### Mathematical Foundation
- In standard transformer attention, the attention scores are computed as:
$$
softmax(q_iK^T)
$$
- ALiBi modifies this to:
$$
softmax(q_iK^T + m\cdot[-(i-1),...,-2,-1,0])
$$
where:
- $q_i \epsilon R^{1Xd}$ Query vector at position $i$ (1<= $i$ <= L)
- $K \epsilon R^{iXd}$ Key matrix containing the first $i$ keys
- $d$ Head dimension
- $m$ is a head-specific slope (fixed before training, not leared) and the bias vector penalizes distant keys linearly. (e.g., $-\frac{1}{2^k}$ for head $k$)
- Bias vector: $[-(i-1),...,-2,-1,0]$ represents distances from query to each key

Slope Computation:
For $n$ attention heads, slopes form a geometric sequence at $2^{-8/n}$ with ratio $2^{-8/n}$.
- 8 heads: $\{\frac{1}{2^1}, \frac{1}{2^2}, ..., \frac{1}{2^8}\}$ 
- 16 heads: $\{\frac{1}{2^1}, \frac{1}{2^2}, ..., \frac{1}{2^{16}}\}$
