#!/bin/sh
set -eu

case "${1:-original}" in
    original)
        model=TakuyaHiraoka/act_so101_pick_diverse_objects/resolve/4cbf28905b71f874a9e10123b01099c0e758bfcf
        dataset=TakuyaHiraoka/so101_pick_diverse_objects/resolve/869b81c2dc4a2b854e029133131b9d8b6332ef7d
        destination=data
        ;;
    compact)
        model=binhpham/sim_so101_cubes_act_small/resolve/9332873218eaba2c64ac1115c4b9742177f64c13
        dataset=binhpham/sim_so101_cubes/resolve/8230a2e80a71dc6254e55bbb1cccfab19e37555d
        destination=data/compact
        ;;
    resnet34)
        model=sancov/so101-ros-act-resnet34/resolve/922a689ce92ec1bcf906174603e6e21536ab7dca
        dataset=sancov/so101-ros-red-ring-merged/resolve/117a91c6a7f053e6617534cfb443879b85ea4b33
        destination=data/resnet34
        ;;
    *)
        echo "Usage: sh fetch-sample.sh [original|compact|resnet34]" >&2
        exit 2
        ;;
esac

model="https://huggingface.co/$model"
dataset="https://huggingface.co/datasets/$dataset"
checkpoint="$destination/checkpoint"
mkdir -p "$checkpoint"

fetch() {
    if [ ! -s "$2" ]; then
        curl --fail --location --silent --show-error --retry 3 "$1" -o "$2.part"
        mv "$2.part" "$2"
    fi
}

for file in config.json policy_preprocessor.json policy_postprocessor.json
do
    fetch "$model/$file" "$checkpoint/$file"
done
processor_states=$(python3 - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path
for processor in Path(sys.argv[1]).glob("policy_*processor.json"):
    for step in json.loads(processor.read_text())["steps"]:
        if "state_file" in step:
            print(step["state_file"])
PY
)
for file in model.safetensors $processor_states
do
    fetch "$model/$file" "$checkpoint/$file"
done

fetch "$dataset/meta/info.json" "$destination/info.json"
fetch "$dataset/data/chunk-000/file-000.parquet" "$destination/frames.parquet"
fetch "$dataset/meta/episodes/chunk-000/file-000.parquet" "$destination/episodes.parquet"
cameras=$(python3 - "$checkpoint/config.json" <<'PY'
import json
import sys
from pathlib import Path
features = json.loads(Path(sys.argv[1]).read_text())["input_features"]
print(*(key for key, feature in features.items() if feature["type"] == "VISUAL"))
PY
)
for camera in $cameras
do
    if [ "$destination/$camera" = "data/observation.images.wrist" ] && [ -s data/wrist.mp4 ]; then
        continue
    fi
    fetch "$dataset/videos/$camera/chunk-000/file-000.mp4" "$destination/$camera.mp4"
done
