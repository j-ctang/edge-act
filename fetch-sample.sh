#!/bin/sh
set -eu
model=https://huggingface.co/TakuyaHiraoka/act_so101_pick_diverse_objects/resolve/4cbf28905b71f874a9e10123b01099c0e758bfcf
dataset=https://huggingface.co/datasets/TakuyaHiraoka/so101_pick_diverse_objects/resolve/869b81c2dc4a2b854e029133131b9d8b6332ef7d
mkdir -p data/checkpoint

fetch() {
    curl --fail --location --silent --show-error --retry 3 "$1" -o "$2.part"
    mv "$2.part" "$2"
}

for file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json \
    policy_postprocessor_step_0_unnormalizer_processor.safetensors \
    policy_preprocessor_step_3_normalizer_processor.safetensors
do
    fetch "$model/$file" "data/checkpoint/$file"
done

fetch "$dataset/data/chunk-000/file-000.parquet" data/frames.parquet
fetch "$dataset/meta/episodes/chunk-000/file-000.parquet" data/episodes.parquet
fetch "$dataset/videos/observation.images.wrist/chunk-000/file-000.mp4" data/wrist.mp4
