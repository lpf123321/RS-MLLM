"""Pinned trusted profile for the 2026 Vision-OPD-9B evaluation checkpoint."""

from __future__ import annotations

from model_policy import TRUSTED_MODEL_PROFILES, TrustedModelProfile


def register_vision_opd_profile() -> TrustedModelProfile:
    profile = TrustedModelProfile(
        key="vision_opd_9b",
        model_ids=frozenset({"yuanqianhao/vision-opd-9b", "vision_opd_9b"}),
        revision="6e41541fc1326b8943ff0b7650db3ce53d6aa826",
        source_url="https://huggingface.co/yuanqianhao/Vision-OPD-9B/tree/6e41541fc1326b8943ff0b7650db3ce53d6aa826",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/yuanqianhao/Vision-OPD-9B/blob/6e41541fc1326b8943ff0b7650db3ce53d6aa826/README.md#license",
        required_files={
            "config.json": 2_900,
            "model.safetensors": 18_819_720_264,
        },
        sha256={
            "config.json": "995196f6106dfbb228e3f198b3eaf14985ef437de88ca5fa5f8910eaf83b2353",
            "model.safetensors": "8727e447b3d42e589672c1b27443a6e8399c92ccfc90ebde3f9cb7e8496d9b9f",
        },
    )
    existing = TRUSTED_MODEL_PROFILES.get(profile.key)
    if existing is not None and existing != profile:
        raise RuntimeError("Conflicting Vision-OPD trusted model profile")
    TRUSTED_MODEL_PROFILES[profile.key] = profile
    return profile
