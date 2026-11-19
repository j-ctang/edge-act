# EdgeACT

Run a pretrained LeRobot ACT policy locally on Apple Silicon and compare PyTorch eager, ExecuTorch with XNNPACK, and Core ML.

## Run the benchmark

Install FFmpeg and uv, then run from the repository root.

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
sh fetch-sample.sh
.venv/bin/python benchmark.py
```

The download pins the [SO-101 ACT checkpoint](https://huggingface.co/TakuyaHiraoka/act_so101_pick_diverse_objects) and matching [recorded demonstrations](https://huggingface.co/datasets/TakuyaHiraoka/so101_pick_diverse_objects). It fetches only the data shards needed for replay. The exports go in `outputs/` and the benchmark writes each measured latency, the sampled frame IDs, and system metadata to `results/benchmark.json`.

## Measurement

Each timed call predicts one action chunk from a batch of one normalized wrist image and joint-state vector. Timings include runtime invocation and output retrieval. Artifact size uses the safetensors weights for PyTorch, the serialized program for ExecuTorch, and all package files for Core ML.

Action MSE and maximum absolute error compare full predicted chunks after the checkpoint's action unnormalization against PyTorch eager. Replay observations come from the source dataset. The experiment measures export parity and inference performance on recorded inputs.

## Results

Apple M5 Pro with 24 GiB RAM, macOS 26.7, on battery with Low Power Mode enabled. Each backend ran 10 warm-ups and 100 timed predictions. Parity used 16 recorded observations from four episodes. PyTorch eager used one CPU thread. Core ML used FP16 conversion with all compute units enabled.

| Backend | p50 (ms) | p99 (ms) | Artifact (MB) | Action MSE | Max absolute action error |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch eager CPU | 106.60 | 114.80 | 206.70 | 0 | 0 |
| ExecuTorch XNNPACK FP32 | 1,313.59 | 1,383.19 | 137.55 | 1.52e-10 | 0.000132 |
| Core ML FP16 | 6.03 | 11.58 | 66.41 | 2.20e-4 | 0.193 |

Core ML had 17.7× lower median latency than PyTorch eager in this run. ExecuTorch's median latency was 12.3× higher than eager despite delegating 20 partitions to XNNPACK. The lowered graph retained CPU fallback operators, including linear algebra and layer normalization. Identifying the latency bottleneck requires operator profiling. Compute-unit placement for Core ML and live task success were outside this measurement.

[Raw benchmark results](results/benchmark.json)
