# FedSALT

Graph-free cross-city traffic forecasting with shared matching and local traffic-state adaptation.

**Python 3.10+ · PyTorch 2.1+ · CPU or GPU · [MIT License](LICENSE)**

FedSALT combines a kernel-based traffic backbone with selective Q/K low-rank adapters and a local state-conditioned FiLM module. The default model has **321,609 parameters**, of which **384 Q/K output-factor parameters** form the sharing interface.

## Quick start

```sh
git clone https://github.com/heart688/FedSALT.git
cd FedSALT
python -m pip install -r requirements.txt
```

Load the supplied configuration and run a forward pass:

```python
import json
import torch
from model import FedSALT

with open("configs/fedsalt.json", encoding="utf-8") as file:
    config = json.load(file)

model = FedSALT(**config["model"])
model.eval()

speed = torch.randn(2, 8, 12)  # [batch, nodes, history]
with torch.no_grad():
    forecast = model(speed)

print(forecast.shape)  # torch.Size([2, 8, 6])
```

This example runs on CPU with random inputs and an untrained model.

## Inputs and outputs

- Input: city-standardized speed, shaped `[batch, nodes, 12]`.
- Output: six future speed values, shaped `[batch, nodes, 6]`.
- Compatibility input: `[batch, nodes, 12, 2]`, with speed in channel 0; channel 1 is unused.

The model needs no graph, node IDs, or calendar features. The node count may vary between inputs.

Window normalization is reversed inside the model. Forecasts remain in **city-standardized units**; apply the corresponding city's inverse normalizer to obtain physical speed values.

## Model configuration

[configs/fedsalt.json](configs/fedsalt.json) separates constructor arguments in `model` from training settings in `local_adaptation_config`. Pass only `config["model"]` to `FedSALT`.

Low-rank adapters act on **Q and K only**. The local `5 → 16 → 256` state conditioner generates FiLM scale and shift, which modulate the common input to Q, K, and V in each block. `model.shared_parameter_names()` returns the six Q/K output-factor matrices used for sharing.

`TrafficBackbone()` provides the source-training backbone. `FedSALT(local_adaptation=False)` provides the collaboration model; the default `FedSALT()` includes the target-local conditioner. Configure each stage before constructing its optimizer.

### Common initialization for collaboration

Following Appendix B.2 of the paper, all clients start from the same source-learned backbone and the same adapter initialization. Each Q/K input factor `A` is drawn from `N(0, 1) / sqrt(dim)`, and each output factor `B` starts at zero. Construct this initialization once, then copy the complete model to the clients:

```python
from copy import deepcopy
from model import FedSALT

# config is loaded as above. source_backbone_state is the state_dict of
# the source-learned TrafficBackbone, after federated source learning.
collaboration_config = {**config["model"], "local_adaptation": False}
initial_model = FedSALT(**collaboration_config)
initial_model.load_source_backbone(source_backbone_state)

# Three source clients followed by the target client (K = 4).
# Each copy has independent parameters with identical initial values.
clients = [deepcopy(initial_model) for _ in range(4)]
collaboration_optimizers = []
for client in clients:
    client.configure_stage("collaboration")
    collaboration_optimizers.append(torch.optim.Adam(
        (p for p in client.parameters() if p.requires_grad),
        lr=0.001,
    ))
```

Creating clients independently and only calling `load_source_backbone()` does not guarantee common `A`: that method retains each instance's existing `A`. Copy the initialization before any client optimization. During collaboration, `A` and the other private parameters subsequently adapt independently; only the Q/K `B` factors are synchronized. The paper delivers the full checkpoint and adapter initialization in FP32 and includes them in its communication accounting. The example above illustrates initialization; federated training and message exchange must be supplied separately.

### Transition from collaboration to target-local adaptation

After collaboration finishes and the target has installed the last decoded `B` broadcast, Appendix B.4 introduces the local `5 → 16 → 256` conditioner. `configure_stage("local")` only changes the trainable mask; it does not add a conditioner to a model constructed with `local_adaptation=False`. Construct the local model and strictly copy the target client's entire post-collaboration backbone, including both adapter factors and all target-private weights:

```python
collaboration_target = clients[-1]  # After the final collaboration round.
target_parameter = next(collaboration_target.parameters())
model = FedSALT(**{**config["model"], "local_adaptation": True}).to(
    device=target_parameter.device,
    dtype=target_parameter.dtype,
)
model.base.load_state_dict(collaboration_target.base.state_dict(), strict=True)
model.configure_stage("local")

# The new conditioner keeps its constructor initialization: a random first
# layer and zero output-layer weights/biases, initially giving identity FiLM.
# Capture once, on the final device/dtype, before any local optimization.
reference = model.capture_value_reference()

settings = config["local_adaptation_config"]
# Fresh Adam; do not reuse the collaboration optimizer or its state.
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=settings["learning_rate"],
)
```

Both adapter factors train locally in this stage. The Q/K base projections, context output projections, and residual scales remain frozen. The paper uses 50 target-local epochs, physical-speed MAE plus the value-anchor penalty below, and no further communication. Do not call `load_source_backbone()` for this transition: it resets `B` and would discard the learned collaboration factors.

## Local value anchoring

Use the detached value-projection reference captured during the transition above. Compute `physical_mae` after city inverse-standardization, then add the anchor penalty at each local training step:

```python
# During each local training step:
loss = physical_mae + model.value_anchor_penalty(
    reference, fit_std,
    coefficient=settings["anchor_coefficient"],
)
```

The supplied local settings use fresh Adam at `0.002` and an anchor coefficient of `0.05`.

### Computing the fixed `fit_std`

In Equation (13) and Appendix B.4, `fit_std` is `sigma_FIT`, the fixed physical-speed standard deviation fitted on the first `floor(0.8 * T_sup)` raw target observations. Here `T_sup` counts the observations available for target training, before making overlapping input/forecast windows:

- Standard support: `T_sup = 864`, so use raw indices `[0:691]` on every city.
- One day on METR-LA or PEMS-BAY: `T_sup = 288`, so use `[0:230]`.
- One day on Chengdu or Shenzhen: `T_sup = 144`, so use `[0:115]`.

For a speed-only tensor `raw_speed` shaped `[time, sensors]` in physical speed units, select the fitting prefix as follows:

```python
T_sup = 864  # Use 288 or 144 for the corresponding one-day setting.
fit_steps = (4 * T_sup) // 5  # Exactly floor(0.8 * T_sup).
raw_speed = torch.as_tensor(raw_speed)
if raw_speed.ndim != 2 or raw_speed.shape[0] < T_sup:
    raise ValueError("Expected physical speeds shaped [time, sensors] covering T_sup.")
fit_speed = raw_speed[:fit_steps]
# Fit the fixed physical-speed standard deviation on this prefix.
```

The paper specifies this fitting range; the model API accepts `fit_std` from the preprocessing pipeline and does not estimate it. Fit this positive, finite scalar once before target-local optimization and keep it fixed throughout that phase. Use only raw speed values, without calendar channels, city standardization, or window normalization. The fit ends at the specified prefix boundary; do not substitute the full support, full-city/test statistics, or a minibatch/window standard deviation. `fit_std` scales the anchor penalty; it is separate from the city normalizer used to invert forecasts into physical units.

The reference is outside `state_dict`. Save and restore the original reference when resuming the same local training phase; capture a new reference only when starting a new adaptation phase.

## Data

Obtain data from the original providers:

- **METR-LA / PEMS-BAY:** [DCRNN authors' data preparation instructions](https://github.com/liyaguang/DCRNN#data-preparation).
- **Chengdu / Shenzhen:** contact the provider through the [DiDi official open-data portal](https://outreach.didichuxing.com/opendata/datasets) for access to the historical traffic-speed data. Availability and access requirements are determined by the provider.

Follow each provider's access terms and prepare the required speed tensors and city normalizers. This repository does not redistribute data.

## Repository

[model.py](model.py) contains the model, [configs/fedsalt.json](configs/fedsalt.json) its configuration, and [requirements.txt](requirements.txt) the dependency.

This release includes model code and configuration; federated orchestration, communication codecs, datasets, and trained checkpoints are not bundled.
