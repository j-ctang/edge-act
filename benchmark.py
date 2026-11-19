"""Export and benchmark a pretrained SO-101 ACT action-chunk predictor."""

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

STATE = "observation.state"
IMAGE = "observation.images.wrist"


class ActionChunk(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, state, *images):
        batch = {STATE: state}
        batch.update(zip(self.policy.config.image_features, images, strict=True))
        return self.policy.predict_action_chunk(batch)


def decode_images(path, ordinals, shape):
    unique = sorted(set(ordinals))
    frame_filter = "+".join(f"eq(n,{frame})" for frame in unique)
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-i", str(path), "-vf", f"select='{frame_filter}'",
        "-frames:v", str(len(unique)), "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-",
    ])
    channels, height, width = shape
    images = np.frombuffer(raw, dtype=np.uint8).reshape(len(unique), height, width, channels)
    by_ordinal = dict(zip(unique, images, strict=True))
    return [by_ordinal[ordinal] for ordinal in ordinals]


def observations(data, preprocess, image_features):
    image_shapes = {key: feature.shape for key, feature in image_features.items()}
    manifest = data / "info.json"
    fps = json.loads(manifest.read_text())["fps"] if manifest.exists() else 30
    frames = pq.read_table(data / "frames.parquet").to_pydict()
    episodes = pq.read_table(data / "episodes.parquet").to_pylist()
    episode_by_id = {episode["episode_index"]: episode for episode in episodes}
    groups = {}
    for row, episode_id in enumerate(frames["episode_index"]):
        episode = episode_by_id[episode_id]
        if all(
            (episode[f"videos/{key}/chunk_index"], episode[f"videos/{key}/file_index"]) == (0, 0)
            for key in image_shapes
        ):
            groups.setdefault(episode_id, []).append(row)
    selected = [
        rows[int(fraction * (len(rows) - 1))]
        for rows in list(groups.values())[:4]
        for fraction in (0.1, 0.4, 0.7, 0.9)
    ]
    if len(selected) != 16:
        raise ValueError("Expected four sampled frames from each of four episodes")

    cameras = {}
    for key, shape in image_shapes.items():
        ordinals = [
            round((episode_by_id[frames["episode_index"][row]][f"videos/{key}/from_timestamp"]
                   + frames["timestamp"][row]) * fps)
            for row in selected
        ]
        path = data / f"{key}.mp4"
        if not path.exists() and key == IMAGE:
            path = data / "wrist.mp4"
        cameras[key] = decode_images(path, ordinals, shape)

    inputs = []
    for idx, row in enumerate(selected):
        batch = preprocess({
            STATE: torch.tensor(frames[STATE][row], dtype=torch.float32),
            **{key: torch.from_numpy(images[idx].copy()).permute(2, 0, 1).float() / 255
               for key, images in cameras.items()},
        })
        inputs.append((batch[STATE], *(batch[key].contiguous() for key in cameras)))
    return inputs, [frames["index"][row] for row in selected]


def benchmark(run, inputs, reference, postprocess, warmups, iterations, size):
    for _ in range(warmups):
        run(*inputs[0])
    times = []
    for index in range(iterations):
        args = inputs[index % len(inputs)]
        start = time.perf_counter_ns()
        run(*args)
        times.append((time.perf_counter_ns() - start) / 1e6)
    errors = torch.stack([
        postprocess(run(*args)) - expected
        for args, expected in zip(inputs, reference, strict=True)
    ]).double()
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p99_ms": float(np.percentile(times, 99)),
        "artifact_bytes": size,
        "action_mse": errors.square().mean().item(),
        "action_max_abs_error": errors.abs().max().item(),
        "latency_ms": times,
    }


def export_executorch(model, example, path):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
    from executorch.runtime import Runtime

    graph = torch.export.export(model, example)
    program = to_edge_transform_and_lower(graph, partitioner=[XnnpackPartitioner()])
    nodes = program.exported_program().graph.nodes
    delegate_target = torch.ops.higher_order.executorch_call_delegate
    delegates = sum(node.target == delegate_target for node in nodes)
    if delegates == 0:
        raise RuntimeError("No model partitions were delegated to XNNPACK")
    details = {
        "xnnpack_partitions": delegates,
        "executorch_fallback_operators": dict(Counter(
            str(node.target) for node in nodes
            if node.op == "call_function" and node.target != delegate_target
        )),
    }
    path.write_bytes(program.to_executorch().buffer)
    method = Runtime.get().load_program(path).load_method("forward")
    return lambda *sample: method.execute(sample)[0], details


def export_coreml(model, example, path, *, compute_units=None):
    import coremltools as ct

    graph = torch.jit.trace(model, example)
    names = ["joint_positions", *(f"image_{i}" for i in range(len(example) - 1))]
    package = ct.convert(
        graph, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
        inputs=[ct.TensorType(name=name, shape=tensor.shape, dtype=np.float32)
                for name, tensor in zip(names, example, strict=True)],
        outputs=[ct.TensorType(name="actions", dtype=np.float32)],
    )
    package.save(str(path))
    package = ct.models.MLModel(str(path), compute_units=compute_units or ct.ComputeUnit.ALL)

    def run(*sample):
        inputs_by_name = {name: tensor.numpy() for name, tensor in zip(names, sample, strict=True)}
        return torch.from_numpy(package.predict(inputs_by_name)["actions"])

    return run


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--results", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--backend", nargs="+", choices=["executorch", "coreml"],
                        default=["executorch", "coreml"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        parser.error("iterations must be positive and warmups nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint = args.data / "checkpoint"
    config = ACTConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True)
    model = ActionChunk(policy).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    inputs, frame_indices = observations(args.data, preprocess, config.image_features)
    reference = [postprocess(model(*sample)) for sample in inputs]
    results = {
        "machine": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "lerobot", "executorch", "coremltools", "pyarrow")},
        "pytorch_threads": torch.get_num_threads(),
        "power": subprocess.check_output(["pmset", "-g", "batt"], text=True).strip(),
        "power_settings": subprocess.check_output(["pmset", "-g", "custom"], text=True).strip(),
        "frame_indices": frame_indices,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "backends": {},
    }

    def record(name, run, size):
        stats = benchmark(run, inputs, reference, postprocess, args.warmups, args.iterations, size)
        stats["power_after_measurement"] = subprocess.check_output(
            ["pmset", "-g"], text=True
        ).strip()
        results["backends"][name] = stats
        print(name, {k: v for k, v in stats.items() if k != "latency_ms"}, flush=True)
        args.results.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

    record("pytorch_cpu", model, (checkpoint / "model.safetensors").stat().st_size)

    if "executorch" in args.backend:
        print("Exporting ExecuTorch model", flush=True)
        path = args.output / "act.pte"
        run, export_details = export_executorch(model, inputs[0], path)
        results.update(export_details)
        record("executorch_xnnpack_fp32", run, path.stat().st_size)

    if "coreml" in args.backend:
        print("Exporting Core ML model", flush=True)
        path = args.output / "act.mlpackage"
        run = export_coreml(model, inputs[0], path)
        size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
        record("coreml_fp16_all", run, size)


if __name__ == "__main__":
    main()
