# EdgeACT

Run a pretrained LeRobot ACT policy locally on Apple Silicon and compare PyTorch eager, ExecuTorch with XNNPACK, and Core ML.

## Run the benchmark

Install FFmpeg and uv, then run from the repository root.

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
sh fetch-sample.sh original
.venv/bin/python benchmark.py --results results/benchmark-ac.json

sh fetch-sample.sh compact
.venv/bin/python benchmark.py --data data/compact --output outputs/compact \
  --results results/benchmark-compact-ac.json

sh fetch-sample.sh resnet34
.venv/bin/python benchmark.py --data data/resnet34 --output outputs/resnet34 \
  --results results/benchmark-resnet34-ac.json
```

The fetch script pins each public checkpoint and its paired LeRobot dataset. It fetches the first data, episode metadata, and video shards required for replay.

| Sample | Existing checkpoint | Recorded inputs | Action chunk |
| --- | --- | --- | ---: |
| Original | [SO-101 diverse-object pick](https://huggingface.co/TakuyaHiraoka/act_so101_pick_diverse_objects) | Wrist, 640 × 480 | 100 |
| Compact | [SO-101 simulated cubes ACT Small](https://huggingface.co/binhpham/sim_so101_cubes_act_small) | Wrist and side, 320 × 240 each | 100 |
| ResNet-34 | [SO-101 ROS ACT ResNet-34](https://huggingface.co/sancov/so101-ros-act-resnet34) | Base and gripper, 640 × 360 each | 50 |

The benchmark writes the sampled frame IDs, raw latencies, system metadata, and output error to the requested results file. Run the checkpoints sequentially to avoid competing workloads.

## Measurement

Each timed call predicts one action chunk from a batch of the normalized camera inputs and joint-state vector. Timings include runtime invocation and output retrieval. Artifact size uses the safetensors weights for PyTorch, the serialized program for ExecuTorch, and all package files for Core ML.

Action MSE and maximum absolute error compare full predicted chunks after the checkpoint's action unnormalization against PyTorch eager. Replay observations come from the source dataset. The experiment measures export parity and inference performance on recorded inputs.

## AC Standard benchmark

All three checkpoints ran on the M5 Pro on AC power in Standard mode. Each backend ran 10 warm-ups and 100 timed calls. Export error compares the full action chunks against the corresponding eager model on 16 recorded observations.

| Checkpoint | Runtime | p50 (ms) | p99 (ms) | Artifact (MB) | Action MSE | Max action error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | PyTorch eager CPU | 74.62 | 115.50 | 206.70 | 0 | 0 |
| Original | ExecuTorch XNNPACK FP32 | 888.68 | 1,024.78 | 137.55 | 1.52e-10 | 0.000132 |
| Original | Core ML FP16 | 4.19 | 5.23 | 66.41 | 0.00022 | 0.193 |
| Compact | PyTorch eager CPU | 34.89 | 36.97 | 62.29 | 0 | 0 |
| Compact | ExecuTorch XNNPACK FP32 | 117.87 | 121.00 | 62.57 | 1.92e-13 | 4.35e-06 |
| Compact | Core ML FP16 | 2.88 | 3.27 | 30.74 | 3.38e-07 | 0.00607 |
| ResNet-34 | PyTorch eager CPU | 149.80 | 155.74 | 246.97 | 0 | 0 |
| ResNet-34 | ExecuTorch XNNPACK FP32 | 2,116.54 | 2,388.23 | 178.07 | 1.11e-10 | 6.87e-05 |
| ResNet-34 | Core ML FP16 | 9.50 | 13.01 | 86.83 | 0.000108 | 0.0575 |

Core ML had the lowest median latency on all three checkpoints. ExecuTorch matched eager more closely on all three and ran slower in each configuration. Model weights, camera count and resolution, and chunk length vary across checkpoints.

[Original raw results](results/benchmark-ac.json) · [Compact raw results](results/benchmark-compact-ac.json) · [ResNet-34 raw results](results/benchmark-resnet34-ac.json)

## Original battery run

Apple M5 Pro with 24 GiB RAM, macOS 26.7, on battery with Low Power Mode enabled. Each backend ran 10 warm-ups and 100 timed predictions. Parity used 16 recorded observations from four episodes. PyTorch eager used one CPU thread. Core ML used FP16 conversion with all compute units enabled.

| Backend | p50 (ms) | p99 (ms) | Artifact (MB) | Action MSE | Max absolute action error |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch eager CPU | 106.60 | 114.80 | 206.70 | 0 | 0 |
| ExecuTorch XNNPACK FP32 | 1,313.59 | 1,383.19 | 137.55 | 1.52e-10 | 0.000132 |
| Core ML FP16 | 6.03 | 11.58 | 66.41 | 2.20e-4 | 0.193 |

Core ML had 17.7× lower median latency than PyTorch eager in this run. ExecuTorch's median latency was 12.3× higher than eager despite delegating 20 partitions to XNNPACK. The lowered graph retained CPU fallback operators, including linear algebra and layer normalization. The AC operator profile below identifies the dominant fallback cost. Compute-unit placement for Core ML and live task success were outside this measurement.

[Raw benchmark results](results/benchmark.json)

## ExecuTorch profiling

Run this after exporting the original checkpoint.

```sh
.venv/bin/python profile_executorch.py
```

On AC Standard power, the original model's ETDump attributed 904.2 ms of summed median operator time to 18 fallback `addmm` calls. The 20 delegated calls accounted for 28.0 ms. A separate macOS stack sample found the fallback kernel in 3,271 of the 3,369 sampled inference frames on the calling thread. These measurements identify the dominant cost in the current ExecuTorch export. [Profile results](results/executorch-profile-ac.json)

## Export parity test

```sh
.venv/bin/python -m unittest discover -s tests
```

The macOS GitHub Actions workflow exercises the same exporters with a deterministic two-camera ACT fixture and checks both output tensors against PyTorch. The Core ML fixture uses CPU-only execution and a 0.01 absolute tolerance in normalized action space. The benchmark enables all compute units.
