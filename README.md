# WaveNet vs MLP vs RNN: Parameter Efficiency Experiments

Author  
\[Your Name/Group\]

Date  
March 28, 2024

Version  
1.1 (Includes RSI+RNN results)

**Acknowledgments:** The core implementation and experimental execution
detailed in this report were performed by \[Your Name/Group\], with
conceptual guidance and orchestration provided by \[AI/Your Name\].

Table of Contents

## Introduction

Standard feedforward neural networks typically rely on linear
transformations (matrix multiplication + bias) followed by simple
non-linear activation functions (e.g., ReLU, Sigmoid). This project
explores an alternative neural network layer construction, termed
"WaveLayer," inspired conceptually by wave function principles, and
evaluates its performance and efficiency against standard Multi-Layer
Perceptron (MLP) and Recurrent Neural Network (RNN) baselines.

Two benchmark tasks were used:

1.  MNIST handwritten digit classification.
2.  Predicting the next day's Relative Strength Index (RSI) for the SPY
    ETF time series.

The primary research question was whether the richer per-connection
function within the WaveLayer could lead to greater parameter efficiency
(achieving comparable accuracy with fewer parameters or higher accuracy
with the same parameters) compared to standard architectures,
particularly on data with inherent periodicity like RSI.

## The WaveLayer Concept

The core idea of the WaveLayer is to replace the simple scalar weight
`W_ij` connecting input `i` to output `j` with a parametric function
incorporating trainable amplitude (`A_ij`), frequency (`ω_ij`), and
phase (`φ_ij`) parameters. The transformation applied to each input
element `x_i` as it contributes to output neuron `j` is given by:

contribution<sub>*i**j*</sub> = *A*<sub>*i**j*</sub> ⋅ sin (*ω*<sub>*i**j*</sub> ⋅ *x*<sub>*i*</sub> + *ϕ*<sub>*i**j*</sub>)

The total input to output neuron `j` before the subsequent activation
function is the sum of these contributions over all inputs `i`, plus a
bias term `B_j`:

output<sub>*j*</sub> = ∑<sub>*i*</sub>\[*A*<sub>*i**j*</sub> ⋅ sin (*ω*<sub>*i**j*</sub> ⋅ *x*<sub>*i*</sub> + *ϕ*<sub>*i**j*</sub>)\] + *B*<sub>*j*</sub>

This "WaveNet" architecture, composed of these WaveLayers, inherently
introduces non-linearity and periodicity *within* the layer
transformation itself.

## Methodology

1.  **Architectures:**
    -   **WaveNet:** Two-layer network:
        `Input -> WaveLayer(In, H) -> Activation -> WaveLayer(H, Out) -> Output`.
        Implemented as `GenericWaveNet` in `models.py`.
    -   **MLP Baseline:** Standard two-layer MLP:
        `Input -> Linear(In, H) -> Activation -> Linear(H, Out) -> Output`.
        Implemented as `GenericMLP` in `models.py`.
    -   **RNN Baselines (RSI only):** Standard `LSTM` and `GRU` layers
        followed by a `Linear` layer for prediction:
        `Input -> LSTM/GRU(features, H) -> Linear(H, Out) -> Output`.
        Implemented as `LstmPredictor` and `GruPredictor` in
        `models.py`.
2.  **Activation Functions:** `ReLU` and `sin` were tested as the
    activation function between the two layers for WaveNet and MLP
    architectures. RNNs used their internal activations.
3.  **Datasets:**
    -   **MNIST:** 60k train / 10k test images (28x28), standard
        normalization. Task: 10-way classification. `Input Size = 784`,
        `Output Size = 10`.
    -   **SPY RSI:** Daily SPY prices (2000-present) used to calculate
        14-period RSI via `pandas_ta`. Task: Predict RSI(t+1) given
        \[RSI(t-13)...RSI(t)\]. 70/15/15 train/val/test split, scaled to
        \[0, 1\]. `Input Size = 14` (sequence length),
        `Output Size = 1`. MultiIndex columns
        <span class="title-ref">('Close', 'SPY')</span> and
        <span class="title-ref">('RSI', 'SPY')</span> handled. Data
        fetched via `data_utils.py`.
4.  **Training:** Models were trained using the Adam optimizer with
    specified learning rates (typically 0.001 or 0.005) and loss
    functions (MSELoss for RSI, CrossEntropyLoss for MNIST). RSI
    training used early stopping based on validation loss (patience=5).
    Experiments were conducted on a **MacBook Air M2 with 8GB RAM**,
    primarily utilizing the CPU.
5.  **Evaluation Metrics:**
    -   MNIST: Test Accuracy (%).
    -   RSI: Root Mean Squared Error (RMSE) on unscaled test
        predictions, compared to a persistence baseline (predict RSI(t)
        = RSI(t-1)).
    -   All: Total number of trainable parameters, total training time
        (where applicable, some runs loaded pre-trained models).
6.  **Implementation:** PyTorch. Code uses external JSON files (e.g.,
    `experiments.json`) for configuring runs and external modules
    (`data_utils.py`, `models.py`).

## Experiment 1: MNIST Classification

### Objective

To establish baseline performance and efficiency of WaveNet vs MLP on a
standard image classification task.

### Key Configurations

A range of hidden sizes were tested initially. The most direct
comparison focused on matching parameter counts:

-   WaveNet (H=24, ~57k params)
-   MLP (H=72, ~57k params)

### MNIST Results

Direct comparison under identical training conditions (LR=0.005, 12
Epochs, CPU):

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

*(Note: A larger WaveNet with H=128 (~305k params) reached ~95%
accuracy, similar to the much smaller MLP H=72).*

### MNIST Conclusion

On the MNIST task, the standard MLP architecture was significantly
**more parameter-efficient** and **computationally faster** than the
WaveNet. The MLP achieved higher accuracy with the same parameter budget
and trained more than twice as fast. The WaveLayer's complexity did not
translate to an advantage on this image classification benchmark.

## Experiment 2: RSI Time Series Prediction

### Objective

To evaluate WaveNet's performance on data with inherent periodicity (RSI
oscillations) and compare it against MLP and standard RNN baselines
(LSTM, GRU).

### Methodology Additions

-   **RNN Models:** LSTM and GRU models (1 layer, 32 hidden units) were
    added as standard time-series baselines.
-   **Parameter Matching:** MLP hidden sizes were adjusted (H=46, H=69)
    to closely match the parameter counts of WaveNet (H=16, H=24)
    respectively.

### RSI Results

Summary of results including parameter-matched MLPs and RNNs:

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
runs. Training time '(loaded)' indicates model state was loaded, not
retrained in final run; estimated times from earlier MNIST/RSI runs
suggest MLP &lt; WaveNet &lt;&lt; RNN)*

### RSI Conclusion

1.  **Baseline:** The persistence baseline was strong; only the best
    models (LSTM, MLP+Sin) achieved a clear improvement.
2.  **Best Accuracy:** The standard `LSTM` achieved the lowest RMSE
    (4.577), confirming its suitability for sequence modeling.
3.  **MLP+Sin Strength:** The `MLP` using a `sin` activation performed
    exceptionally well, nearly matching the LSTM's accuracy (RMSE 4.586)
    but with significantly fewer parameters (~1.1k vs ~4.5k) and likely
    much faster training.
4.  **WaveNet Performance:** WaveNet performed reasonably (RMSEs
    ~4.61-4.64), slightly beating the baseline, but was consistently
    outperformed by the MLP+Sin architecture at equivalent parameter
    counts.
5.  **Parameter Efficiency:** The `MLP+Sin` architecture demonstrated
    the best parameter efficiency among models clearly beating the
    baseline. WaveNet was less efficient. LSTM achieved top accuracy at
    the highest parameter cost.
6.  **Activation:** `sin` activation was crucial for MLP performance on
    RSI, far exceeding `ReLU`.

Overall, for RSI prediction, LSTM was most accurate, while **MLP+Sin
offered the best balance of accuracy, parameter efficiency, and speed**.
The WaveNet, despite its periodic bias, was not the most effective or
efficient architecture.

## Overall Discussion

### Efficiency Comparison Summary

Across both MNIST classification and RSI time-series prediction, the
proposed WaveNet architecture, while functional, consistently
demonstrated **lower parameter efficiency** and **higher computational
cost** compared to standard MLP baselines. On MNIST, the MLP achieved
higher accuracy for the same parameters. On RSI, the MLP+Sin
configuration achieved better accuracy for the same parameters compared
to WaveNet.

### Hypothesis on Periodic Data

The initial hypothesis that WaveNet's inherent periodicity might give it
an advantage on oscillating data like RSI was **not supported** by the
results. While the MLP's *relative* advantage over WaveNet (seen on
MNIST) diminished on RSI, WaveNet did not become superior. The simpler
**MLP+Sin architecture proved more parameter-efficient than WaveNet** on
the RSI task.

### The "Wave" Inspiration and Reality

The WaveLayer design was conceptually inspired by using wave-like
functions for network transformations, potentially capturing complex
patterns more efficiently, echoing ideas from signal processing or
physics (e.g., wave-particle duality rhyme).

However, the experimental results suggest this conceptual appeal did not
translate into a practical advantage for these tasks and this specific
implementation. Potential reasons include:

-   **Optimization Challenges:** Training multiple parameters (A, ω, φ)
    per connection might be harder than training single weights.
-   **Complexity Overhead:** The computation within WaveLayer is likely
    much slower than optimized matrix multiplications.
-   **Bias Mismatch/Task Difficulty:** The specific
    <span class="title-ref">A\*sin(ωx+φ)</span> form might not be
    optimal, or other signal characteristics (noise) dominate. The RSI
    task itself proved difficult for most models to significantly beat
    the persistence baseline.

### Effectiveness of MLP+Sin

An interesting finding was the effectiveness of a standard MLP using a
`sin` activation function, particularly on RSI. This simple modification
significantly outperformed the `ReLU` MLP and was more
parameter-efficient than WaveNet, suggesting that introducing
periodicity via the activation function is a valuable and efficient
technique for certain time-series modeling tasks.

## Overall Conclusion

This study implemented and evaluated a novel "WaveNet" architecture
using layers based on parametric wave-like functions. Comparative
experiments on MNIST classification and RSI time-series prediction
against MLP and RNN baselines (run on a MacBook Air M2 CPU) showed that:

1.  The WaveNet architecture is functional and capable of learning.
2.  However, on both tasks, WaveNet was found to be **less
    parameter-efficient** and **computationally slower** than standard
    MLP baselines.
3.  Specifically on the RSI task, the **MLP+Sin architecture
    demonstrated superior parameter efficiency** compared to WaveNet,
    achieving better accuracy with the same parameter budget.
4.  While standard LSTM models achieved the highest accuracy on RSI,
    MLP+Sin offered a compelling balance of accuracy, parameter count,
    and computational speed.

The results suggest that, for the problems and configurations tested,
the additional complexity introduced by the WaveLayer's per-connection
parametric wave function does not provide a net benefit over simpler,
well-established neural network components, even on potentially
favorable periodic data.

## Code Availability

The PyTorch code used for these experiments, allowing configuration via
JSON files and replication of WaveNet, MLP, LSTM, and GRU models, is
available in this repository: \[Link to your Repository if applicable\]
