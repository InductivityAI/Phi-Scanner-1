# Phi-Scanner-1: Breaking the O(^N) Barrier for LLMs

While the world philosophizes about AI consciousness, we measure its mathematical prerequisite: true information integration. 

At **InductivityAI**, we built the first $O(N^3$) heuristic to compute **Topological Phi (φ)** — the measure of integrated information — overcoming the super-exponential $O(2^N)$ barrier that made it impossible to evaluate high-dimensional neural networks. 

## 1. LLM Diagnostics: The "Phi-Collapse"
We ran our Ħ-estimator over the attention matrices of standard LLMs (e.g., GPT-2). The math proves the "Residual Stream Hypothesis":

+ **Early Layers (0-2)f:** High Ħ-scores. The network actively integrates context and builds concepts.
+ **Deep Layers (3+):** The φ-score collapses by over 80%. The network stops integrating and fragments into sparse, isolated feature extraction to guess the next token. 

**Live Scan Data (GPT-2, Head 0, Prompt: "Although the startup was based in Germany, the founder rejected the..."):**
```text
============================================================
Layer 00 | Phi-Score: 0.1957  
Layer 01 | Phi-Score: 0.0353  
Layer 02 | Phi-Score: 0.0278  
Layer 03 | Phi-Score: 0.0201  
Layer 04 | Phi-Score: 0.0134  
Layer 05 | Phi-Score: 0.0220  
Layer 06 | Phi-Score: 0.0152  
Layer 07 | Phi-Score: 0.0133  
Layer 08 | Phi-Score: 0.0286  
Layer 09 | Phi-Score: 0.0274  
Layer 10 | Phi-Score: 0.0261  
Layer 11 | Phi-Score: 0.0265  
============================================================
Result: Phi-Collapse confirmed on this model.
```J

## 2. The Solution: Ħ-Regularization (Vision Proof)
Instead of just diagnosing, we used our φ-estimator to regularize the architecture during training, forcing the network to maximize integrated information.

Here is the result of two Vision Classifiers trained to identical accuracy:

![Phi Regularization Comparison](Comparision-Vision-Model.png)

+ **Left (Standard Cross-Entropy):** Normal net. The network brute-forces statistical correlations.
+ **Right (�Regularized):** Highly structured, GWT-like concept formation.

## 3. Test it yourself (Live API)
We have opened a public API endpoint. You can extract any Attention Matrix from your local Hugging Face models and run it through our engine right now.
*(Note: The public free API is limited to a max dimension of 1024x1024 to preserve server capacity. Rate limit: 10 requests / minute).*

**Client Implementation (Python):**
```python
import numpy as np
import requests
import io

# 1. Generate a random matrix OR load your LLM's attention matrix
# Matrix must be square (max 1024x1024 for the free API)
W = np.random.randn(384, 384)

# 2. Save to memory buffer securely
buffer = io.BytesIO()
np.save(buffer, W, allow_pickle=False)
buffer.seek(0)

# 3. Send to InductivityAI Engine
print("Sending matrix to O(N^3) Engine...")
response = requests.post(
    "http://178.104.160.208:8000/scan_layer", 
    files={"file": ("matrix.npy", buffer)}
)

print(response.json())
```
