"""Capture steady-state ExecuTorch operator timings for a deployed ACT model."""

import argparse
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import torch
from executorch.devtools import Inspector
from executorch.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=Path("outputs/act.pte"))
    parser.add_argument("--config", type=Path, default=Path("data/checkpoint/config.json"))
    args = parser.parse_args()

    features = json.loads(args.config.read_text())["input_features"]
    shapes = [
        features["observation.state"]["shape"],
        *(feature["shape"] for feature in features.values() if feature["type"] == "VISUAL"),
    ]
    torch.set_num_threads(1)
    inputs = [torch.zeros(1, *shape) for shape in shapes]
    program = Runtime.get().load_program(args.program, enable_etdump=True)
    method = program.load_method("forward")
    for _ in range(9):
        method.execute(inputs)

    with TemporaryDirectory() as temporary:
        etdump = str(Path(temporary) / "profile.etdp")
        debug = str(Path(temporary) / "profile.debug")
        program.write_etdump_result_to_file(etdump, debug)
        events = Inspector(etdump_path=etdump).to_dataframe()
    execution = events[events["event_block_name"] == "Execute"].copy()
    # The first invocation warms the runtime. Exclude it from every event series.
    execution["median_ms"] = execution["raw"].map(
        lambda calls: median(calls[1:])
    )
    summary = execution.groupby("event_name")["median_ms"].agg(["size", "sum"])
    summary = summary.sort_values("sum", ascending=False)
    print("Sum of median event durations in milliseconds. Parent and child events overlap.")
    print(summary.head(20).to_string())


if __name__ == "__main__":
    main()
