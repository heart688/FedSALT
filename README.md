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

`TrafficBackbone()` provides the source-training backbone. `FedSALT(local_adaptation=False)` provides the collaboration model; the default `FedSALT()` enables local adaptation. Configure the stage before constructing an optimizer:

```python
model.configure_stage("local")
settings = config["local_adaptation_config"]
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=settings["learning_rate"],
)
```

For collaboration, construct the model with `local_adaptation=False` and use `configure_stage("collaboration")`.

## Local value anchoring

After collaboration, move the model to its intended device and dtype, then capture the value-projection weights and biases once before local optimization:

```python
reference = model.capture_value_reference()
# During each local training step:
loss = physical_mae + model.value_anchor_penalty(
    reference, fit_std,
    coefficient=settings["anchor_coefficient"],
)
```

Compute `physical_mae` after city inverse-standardization. `fit_std` is the positive physical-speed standard deviation fitted on the designated target support prefix. The supplied local settings use Adam, a learning rate of `0.002`, and an anchor coefficient of `0.05`.

The reference is outside `state_dict`. Save and restore the original reference when resuming the same local training phase; capture a new reference only when starting a new adaptation phase.

## Data

Obtain data from the original providers:

- **METR-LA / PEMS-BAY:** [DCRNN authors' data preparation instructions](https://github.com/liyaguang/DCRNN#data-preparation).
- **Chengdu / Shenzhen:** contact the provider through the [DiDi official open-data portal](https://outreach.didichuxing.com/opendata/datasets) for access to the historical traffic-speed data. Availability and access requirements are determined by the provider.

Follow each provider's access terms and prepare the required speed tensors and city normalizers. This repository does not redistribute data.

## Repository

[model.py](model.py) contains the model, [configs/fedsalt.json](configs/fedsalt.json) its configuration, and [requirements.txt](requirements.txt) the dependency.

This release includes model code and configuration; federated orchestration, communication codecs, datasets, and trained checkpoints are not bundled.
