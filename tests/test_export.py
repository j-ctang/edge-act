"""Check export parity on a tiny, deterministic two-camera ACT policy."""

import tempfile
import unittest
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from benchmark import ACTConfig, ACTPolicy, ActionChunk, STATE, export_coreml, export_executorch


class ExportParityTest(unittest.TestCase):
    @torch.no_grad()
    def test_both_runtimes_match_eager(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        image_shape = (3, 32, 32)
        config = ACTConfig(
            device="cpu",
            input_features={
                STATE: PolicyFeature(FeatureType.STATE, (6,)),
                "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, image_shape),
                "observation.images.side": PolicyFeature(FeatureType.VISUAL, image_shape),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
            pretrained_backbone_weights=None,
            chunk_size=3,
            n_action_steps=3,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            use_vae=False,
        )
        model = ActionChunk(ACTPolicy(config)).eval()
        samples = [
            (torch.rand(1, 6), torch.rand(1, *image_shape), torch.rand(1, *image_shape))
            for _ in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executorch, _ = export_executorch(model, samples[0], root / "act.pte")
            for sample in samples:
                torch.testing.assert_close(executorch(*sample), model(*sample), atol=1e-5, rtol=1e-4)
            coreml = export_coreml(model, samples[0], root / "act.mlpackage")
            for sample in samples:
                torch.testing.assert_close(coreml(*sample), model(*sample), atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    unittest.main()
