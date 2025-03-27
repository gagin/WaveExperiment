# WaveNet vs MLP vs RNN: Parameter Efficiency Experiments

Author  
Alex Gaggin

Date  
March 28, 2024

Version  
1.2 (Includes RNN baselines, highlights MLP+Sin)

**Acknowledgments:** The core implementation and experimental execution
detailed in this report were performed by Gemini 2.5 Pro Experimental
03-25, with conceptual guidance and orchestration provided by the
author.

Table of Contents

## Introduction

Standard feedforward neural networks typically rely on linear
transformations followed by simple non-linear activation functions
(e.g., ReLU). This project explores an alternative layer construction,
termed "WaveLayer," inspired by wave function concepts, evaluating its
performance and efficiency against standard Multi-Layer Perceptron (MLP)
and Recurrent Neural Network (RNN) baselines on MNIST classification and
SPY RSI time-series prediction.

The primary research question was whether the WaveLayer's richer
per-connection function could yield greater parameter efficiency. While
investigating this, the study also uncovered significant insights into
the effectiveness of using periodic activation functions
(<span class="title-ref">sin</span>) within standard MLP architectures,
particularly for time-series data.

## The WaveLayer Concept

The core idea of the WaveLayer is to replace the scalar weight `W_ij`
with a parametric function incorporating trainable amplitude (`A_ij`),
frequency (`ω_ij`), and phase (`φ_ij`):

contribution<sub>*i**j*</sub> = *A*<sub>*i**j*</sub> ⋅ sin (*ω*<sub>*i**j*</sub> ⋅ *x*<sub>*i*</sub> + *ϕ*<sub>*i**j*</sub>)

The total pre-activation input to output neuron `j` is:

output<sub>*j*</sub> = ∑<sub>*i*</sub>\[*A*<sub>*i**j*</sub> ⋅ sin (*ω*<sub>*i**j*</sub> ⋅ *x*<sub>*i*</sub> + *ϕ*<sub>*i**j*</sub>)\] + *B*<sub>*j*</sub>

This "WaveNet" architecture introduces non-linearity and periodicity
*within* the layer transformation.

## Methodology

1.  **Architectures:**
    -   **WaveNet:** Two-layer `GenericWaveNet` (`models.py`).
    -   **MLP Baseline:** Two-layer `GenericMLP` (`models.py`).
    -   **RNN Baselines (RSI only):** `LstmPredictor` and `GruPredictor`
        (`models.py`).
2.  **Activation Functions:** `ReLU` and `sin` tested between layers for
    WaveNet/MLP.
3.  **Datasets:**
    -   **MNIST:** Standard setup (784 input -&gt; 10 output).
    -   **SPY RSI:** 14-day RSI prediction (14 input -&gt; 1 output),
        MultiIndex handled, data via `data_utils.py`.
4.  **Training:** Adam optimizer, relevant loss functions
    (MSE/CrossEntropy), early stopping for RSI. Hardware: **MacBook Air
    M2, 8GB RAM** (CPU).
5.  **Evaluation Metrics:** MNIST Accuracy (%), RSI RMSE (vs.
    persistence baseline), parameter counts, training time.
6.  **Implementation:** PyTorch, configurable via JSON (e.g.,
    `experiments.json`), modular code (`data_utils.py`, `models.py`).

## Experiment 1: MNIST Classification

### Objective

Establish baseline performance and efficiency of WaveNet vs MLP on image
classification.

### Key Configurations

Focus on parameter-matched comparison: WaveNet (H=24, ~57k params) vs.
MLP (H=72, ~57k params).

### MNIST Results

<table>
<caption>MNIST Parameter Efficiency Comparison</caption>
<thead>
<tr>
<th>Model Configuration</th>
<th>Parameters</th>
<th>Epochs</th>
<th>Test Accuracy</th>
<th>Approx. Train Time (s)</th>
</tr>
</thead>
<tbody>
<tr>
<td>WaveNet (H=24, sin)</td>
<td><blockquote>
<p>57,202</p>
</blockquote></td>
<td><blockquote>
<p>12</p>
</blockquote></td>
<td><blockquote>
<p>93.54%</p>
</blockquote></td>
<td><blockquote>
<p>~86s</p>
</blockquote></td>
</tr>
<tr>
<td>MLP (H=72, ReLU)</td>
<td><blockquote>
<p>57,250</p>
</blockquote></td>
<td><blockquote>
<p>12</p>
</blockquote></td>
<td><blockquote>
<p><strong>95.85%</strong></p>
</blockquote></td>
<td><blockquote>
<p><strong>~40s</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

*(Note: Larger WaveNet H=128 (~305k params) required to reach ~95%
accuracy).*

### MNIST Conclusion

On MNIST, the standard **MLP was significantly more parameter-efficient
and computationally faster** than WaveNet, achieving higher accuracy
with the same parameters.

## Experiment 2: RSI Time Series Prediction

### Objective

Evaluate WaveNet on periodic data (RSI) against MLP (parameter-matched)
and standard RNN baselines.

### Methodology Additions

-   RNN models (LSTM H=32, GRU H=32) added.
-   MLP hidden sizes adjusted (H=46, H=69) for accurate parameter
    matching against WaveNet (H=16, H=24).

### RSI Results

<table>
<caption>RSI Prediction Experiment Summary</caption>
<tbody>
<tr>
</tr>
<tr>
</tr>
<tr>
</tr>
<tr>
</tr>
<tr>
</tr>
<tr>
</tr>
<tr>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>

*(Note: Baseline RMSE ~4.6425. WaveNet results included from previous
runs. Training time '(loaded)' indicates previous state loaded)*

### RSI Conclusion

1.  **Baselines:** LSTM achieved the best accuracy (RMSE 4.577). The
    persistence baseline (RMSE ~4.64) remained challenging.
2.  **MLP+Sin Strength:** The `MLP` using `sin` activation emerged as a
    highly effective architecture, nearly matching LSTM accuracy (RMSE
    4.586) with significantly fewer parameters (~1.1k vs ~4.5k) and
    faster expected training time.
3.  **WaveNet vs MLP+Sin:** WaveNet was consistently outperformed by
    MLP+Sin at equivalent parameter counts in terms of accuracy.
4.  **Parameter Efficiency:** The **\`\`MLP+Sin\`\` architecture offered
    the best balance of accuracy and parameter efficiency**. LSTM was
    most accurate but least efficient. WaveNet was less efficient than
    MLP+Sin.
5.  **Activation:** `sin` activation was crucial for good MLP
    performance on RSI, significantly outperforming `ReLU`.

## Overall Discussion

### Efficiency Comparison Summary

Across both tasks, the custom **WaveNet architecture was less
parameter-efficient and computationally slower than standard MLPs**. On
RSI, the MLP's advantage was most pronounced when using a `sin`
activation.

### Hypothesis on Periodic Data

The hypothesis that WaveNet's periodic bias would be advantageous on RSI
data was **not supported**. The simpler MLP+Sin architecture proved more
parameter-efficient and achieved higher accuracy than WaveNet at matched
parameter counts.

### The "Wave" Inspiration and Reality

While conceptually appealing, drawing inspiration from wave-like
functions, the practical implementation of WaveLayer faced challenges.
Its complexity likely led to optimization difficulties and computational
overhead that outweighed potential benefits from its inductive bias for
the tasks tested.

### Effectiveness of MLP+Sin - A Key Finding

A significant outcome of this investigation was the **demonstrated
effectiveness of using a standard MLP architecture with a \`\`sin\`\`
activation function for the periodic RSI time series**. This relatively
simple approach (MLP+Sin) achieved performance close to the best LSTM
model but with vastly superior parameter and computational efficiency
compared to both LSTM and WaveNet. This highlights a practical and
efficient method for incorporating periodic bias into models for
relevant tasks.

## Overall Conclusion

This study evaluated a novel "WaveNet" architecture using parametric
wave functions. Experiments on MNIST and RSI prediction (vs MLP, LSTM,
GRU baselines on a MacBook Air M2 CPU) led to two main conclusions:

1.  The custom **WaveNet architecture, while functional, proved less
    parameter-efficient and computationally slower** than standard MLP
    baselines on both tasks. Its inherent periodic bias did not
    translate into a competitive advantage, even on the oscillating RSI
    data.
2.  A key secondary finding was the **high effectiveness and efficiency
    of using a simple \`sin\` activation function within a standard
    MLP** for the RSI time-series task. This MLP+Sin configuration
    offered a superior balance of accuracy, parameter count, and speed
    compared to WaveNet, LSTM, and standard ReLU MLPs for this specific
    problem.

The results suggest that the added complexity of the WaveLayer did not
yield practical benefits over simpler, established methods, while also
highlighting the potential of using periodic activation functions in
standard networks for time-series modeling.

## Code Availability

The PyTorch code used for these experiments, allowing configuration via
JSON files and replication of WaveNet, MLP, LSTM, and GRU models, is
available in this repository.
