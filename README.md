# WaveNet vs MLP: Parameter Efficiency on MNIST Classification

::: {.contents local="" depth="2"}
Table of Contents
:::

## Introduction

Standard feedforward neural networks typically rely on linear
transformations (matrix multiplication plus bias) followed by simple
non-linear activation functions (e.g., ReLU, Sigmoid). This project
explores an alternative neural network layer construction, termed
\"WaveLayer,\" inspired by wave function concepts, and evaluates its
performance and efficiency against a standard Multi-Layer Perceptron
(MLP) baseline on the MNIST handwritten digit classification task.

The core idea of the WaveLayer is to replace the simple scalar weight
`W_ij` connecting input `i` to output `j` with a parametric function
incorporating trainable amplitude (`A_ij`), frequency (`ω_ij`), and
phase (`φ_ij`) parameters. The transformation applied to each input
element `x_i` as it contributes to output neuron `j` is given by:

$$\text{contribution}_{ij} = A_{ij} \cdot \sin(\omega_{ij} \cdot x_i + \phi_{ij})$$

The total input to output neuron `j` before the subsequent activation
function is the sum of these contributions over all inputs `i`, plus a
bias term `B_j`:

$$\text{output}_j = \sum_{i} [ A_{ij} \cdot \sin(\omega_{ij} \cdot x_i + \phi_{ij}) ] + B_j$$

This \"WaveNet\" architecture, composed of these WaveLayers, inherently
introduces non-linearity and periodicity *within* the layer
transformation itself. The primary research question was whether this
richer per-connection function could lead to greater parameter
efficiency (achieving comparable accuracy with fewer parameters)
compared to a standard MLP.

## Methodology

1.  **Architecture:**
    -   **WaveNet:** A two-layer network consisting of
        `Input -> WaveLayer(784, H) -> Activation -> WaveLayer(H, 10) -> Output`.
    -   **MLP Baseline:** A standard two-layer MLP:
        `Input -> Linear(784, H) -> Activation -> Linear(H, 10) -> Output`.
2.  **Activation Functions:** Both `ReLU` and `sin` were tested as the
    activation function between the two layers for both WaveNet and MLP
    architectures.
3.  **Dataset:** MNIST (60,000 training images, 10,000 test images).
    Standard normalization was applied.
4.  **Training:** Models were trained using the Adam optimizer with
    varying learning rates (0.005, 0.001, 0.0005) for a specified number
    of epochs (typically 12-15, sometimes up to 30). Training was
    performed on a CPU.
5.  **Evaluation Metrics:**
    -   Test Accuracy (%) on the MNIST test set.
    -   Total number of trainable parameters.
    -   Total training time (seconds).
6.  **Implementation:** PyTorch. The code provided in this repository
    implements both architectures and allows configuration via
    parameters.

## Experiments and Key Results

A series of experiments were conducted, varying the model type
(WaveNet/MLP), hidden layer size (H), activation function, learning
rate, and training epochs.

1.  **WaveNet Viability:** Initial tests confirmed that the WaveNet
    architecture *can* learn the MNIST task, with larger models (e.g.,
    H=128) achieving competitive accuracy (\~95%).

2.  **Parameter Reduction in WaveNet:** The hidden size (H) for WaveNet
    was reduced to investigate parameter efficiency:

    -   [H=24]{.title-ref} (\~57k parameters) emerged as a point of
        interest, achieving **\~94.2%** test accuracy after 15 epochs
        (using [sin]{.title-ref} activation and LR=0.005).
    -   Further reductions ([H=16]{.title-ref} \~38k params,
        [H=8]{.title-ref} \~19k params) led to significant drops in
        accuracy (to \~91.8% and \~88.7% respectively), suggesting a
        lower limit for representational capacity.

3.  **Activation Function Impact (WaveNet H=24):**

    -   Using [sin]{.title-ref} activation between WaveLayers resulted
        in faster initial convergence compared to [ReLU]{.title-ref}
        (93.8% after 3 epochs vs. \~91.8% for ReLU after 5 epochs).
    -   However, after sufficient training (15 epochs), both
        [sin]{.title-ref} and [ReLU]{.title-ref} versions converged to
        the **same peak accuracy of \~94.2%**.

4.  **Direct Baseline Comparison (Matched Parameters & Training):** The
    most critical comparison was between WaveNet (H=24, \~57k params)
    and MLP (H=72, \~57k params), trained under identical conditions
    (LR=0.005, 12 Epochs):

    +--------------------+---------+---------+-------------+-----------+
    | Model              | Par     | Epochs  | Test        | Train     |
    | Configuration      | ameters |         | Accuracy    | Time (s)  |
    +====================+=========+=========+=============+===========+
    | WaveNet (H=24,     | >       | > 12    | > 93.54%    | > \~86s   |
    | sin)               |  57,202 |         |             |           |
    +--------------------+---------+---------+-------------+-----------+
    | MLP (H=72, ReLU)   | >       | > 12    | >           | >         |
    |                    |  57,250 |         |  **95.85%** | **\~40s** |
    +--------------------+---------+---------+-------------+-----------+

    -   The standard MLP achieved significantly **higher accuracy**
        (\>2.3 points) with the same parameter budget.
    -   The standard MLP trained **more than twice as fast** due to the
        computational efficiency of linear layers versus the
        WaveLayer\'s complex operations.

5.  **Optimization Challenges:** Attempts to improve WaveNet performance
    by increasing hidden size (e.g., H=32) or extending training time
    with the initial LR (0.005) did not surpass the H=24 peak accuracy,
    suggesting potential optimization difficulties or the need for
    careful LR tuning and regularization for larger WaveNets.

## Discussion

The experiments demonstrate that while the proposed WaveLayer concept is
functional and allows the network to learn complex patterns, it does not
offer clear advantages over standard MLP architectures for the MNIST
task in terms of parameter efficiency or computational speed on
conventional hardware.

-   **Efficiency:** At matched parameter counts (\~57k), the standard
    MLP achieved higher accuracy. The WaveNet required significantly
    more parameters (H=128, \~305k) to reach similar (\>95%) accuracy
    levels as moderately sized MLPs.
-   **Computational Cost:** The calculation involving [sin]{.title-ref}
    and multiple parameter interactions per connection within the
    WaveLayer incurs substantial computational overhead, leading to
    significantly longer training times compared to highly optimized
    matrix multiplications in linear layers.
-   **Potential Niche:** The inductive bias towards periodic functions
    introduced by the WaveLayer might be beneficial for datasets with
    strong inherent periodicity (e.g., signal processing, time series
    with clear cycles), but this was not evident for MNIST image
    classification.

## Conclusion

This study implemented and evaluated a novel \"WaveNet\" architecture
using layers based on parametric wave-like functions for MNIST digit
classification. While demonstrating the viability of such custom layers,
the results indicate that, for this benchmark task and on standard CPU
hardware, the WaveNet is less parameter-efficient and computationally
slower than a standard MLP baseline achieving comparable or superior
accuracy. The H=24 WaveNet configuration (\~57k parameters) reached a
peak accuracy of \~94.2%, falling short of an equivalently sized MLP
baseline (\~95.9%). Future work could explore the WaveNet\'s
applicability to datasets with stronger periodic characteristics or
investigate performance on specialized hardware architectures.

## Code Availability

The PyTorch code used for these experiments, allowing replication and
further exploration of both WaveNet and MLP models, is available in this
repository. Configuration parameters (model type, hidden size,
activation, LR, epochs) can be adjusted within the main script.
