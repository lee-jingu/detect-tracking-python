from __future__ import annotations

import torch

from models.yolor import Darknet


def get_darknet(model_config: str, img_size: tuple[int, int] | int,
                device: str, weight_path: str) -> Darknet:
    model = Darknet(model_config, img_size).to(device)
    model.load_state_dict(
        torch.load(weight_path, map_location=device)["model"])
    return model
