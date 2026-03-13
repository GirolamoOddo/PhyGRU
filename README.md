



### PhyGRU - A Physics-Biased Variant for Gated Recurrent Unit
**Companion Preprint:** Available at https://doi.org/10.5281/zenodo.18911329


---

#### Overwiew

Modeling dynamical systems with partial physical knowledge and limited data is challenging. Purely data-driven recurrent networks can struggle to generalize, while simplified physical models often fail to capture the full system dynamics. PhyGRU addresses this by embedding an explicit physics-informed candidate state within a standard Gated Recurrent Unit architecture. 

In PhyGRU, the candidate state is obtained by forward integrating a parametric physical model, optionally complemented by a low-dimensional latent state that captures unmodeled dynamics. The standard GRU update gate is preserved, allowing smooth blending between the previous state and the physics-informed candidate. Unlike conventional GRUs, the reset gate is removed for the physical component to maintain continuity and physical consistency.

This design enables interpretable, physics-guided temporal propagation while maintaining the flexibility of recurrent neural networks. PhyGRU requires only minimal changes to a standard GRU, making it easy to integrate into existing pipelines. The approach is particularly suited to scenarios where partial physical knowledge exists and fully data-driven models alone are insufficient, offering a principled way to combine prior knowledge with learned corrections.

---

#### Installation

Clone the repository and install dependencies using `pip`:

```bash
git clone https://github.com/GirolamoOddo/PhyGRU.git
cd PhyGRU
pip install -r requirements.txt
```
The file `PhyGRU_example.py` provides a ready-to-run example demonstrating how to use PhyGRU in a PyTorch environment. It shows how to initialize an hybrid PhyGRU-GRU-FNN layers model, define input sequences, perform forward passes, and inspect outputs. A MATLAB R2021b+ with Deep Learning Toolbox version is also provided. The data in the examples are randomly generated solely to illustrate the workflow; users can customize the data and model structure for their specific applications.

