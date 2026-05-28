from pathlib import Path

import torch

from app.inference.class_mapping import driver4_raw_class_names
from app.inference import model_loader
from app.inference.model_loader import LoadedModel


def _fake_loaded_model(model_path: Path, compiled: bool) -> LoadedModel:
    return LoadedModel(
        model=torch.nn.Identity(),
        class_names=[model_path.name],
        device=torch.device("cpu"),
        model_path=model_path,
        compiled=compiled,
        architecture="test",
    )


def test_model_loader_keeps_only_one_active_model(monkeypatch, tmp_path) -> None:
    first_model_path = tmp_path / "first.pth"
    second_model_path = tmp_path / "second.pth"
    first_model_path.touch()
    second_model_path.touch()
    loads: list[Path] = []
    releases = 0

    def fake_load_model_from_path_uncached(model_path: Path, compiled: bool = False) -> LoadedModel:
        loads.append(model_path)
        return _fake_loaded_model(model_path, compiled)

    def fake_release_cached_models() -> None:
        nonlocal releases
        releases += 1
        model_loader._active_model_cache.clear()

    monkeypatch.setattr(model_loader, "_load_model_from_path_uncached", fake_load_model_from_path_uncached)
    monkeypatch.setattr(model_loader, "_release_cached_models", fake_release_cached_models)
    model_loader._active_model_cache.clear()

    first_loaded_model = model_loader.load_model_from_path(first_model_path)
    first_cached_model = model_loader.load_model_from_path(first_model_path)
    second_loaded_model = model_loader.load_model_from_path(second_model_path)

    assert first_loaded_model is first_cached_model
    assert second_loaded_model.model_path == second_model_path.resolve()
    assert loads == [first_model_path.resolve(), second_model_path.resolve()]
    assert releases == 1
    assert list(model_loader._active_model_cache.keys()) == [(second_model_path.resolve(), False)]

    model_loader._active_model_cache.clear()


def test_timm_four_class_checkpoint_loads_as_driver4_model(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "final_model_0528.pth"
    torch.save(
        {
            "model_state_dict": {
                "backbone.patch_embed.proj.weight": torch.zeros(1),
                "backbone.head.weight": torch.zeros(len(driver4_raw_class_names), 768),
                "backbone.head.bias": torch.zeros(len(driver4_raw_class_names)),
            },
            "class_names": list(driver4_raw_class_names),
        },
        model_path,
    )

    class FakeTimmModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes

        def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
            self.loaded_state_dict = state_dict
            self.strict = strict
            return torch.nn.modules.module._IncompatibleKeys([], [])

    monkeypatch.setattr(model_loader, "BamtiTimmVisionModel", FakeTimmModel)

    loaded_model = model_loader.load_model_from_path(model_path)

    assert loaded_model.architecture == "timm_vit_b_16_driver4"
    assert loaded_model.raw_class_names == driver4_raw_class_names
    assert [service_class.variable_name for service_class in loaded_model.service_classes] == [
        "body_touching",
        "distraction",
        "phone_operation",
        "steering_operation",
    ]
    assert loaded_model.model.num_classes == 4
    assert loaded_model.class_names == [
        "body_touching",
        "distraction",
        "phone_operation",
        "steering_operation",
    ]

    model_loader._active_model_cache.clear()
