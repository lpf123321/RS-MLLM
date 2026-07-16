from pathlib import Path
import sys
from types import SimpleNamespace

import torch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from candidate_registry import CANDIDATES
from eval_candidates import (
    ADAPTERS,
    AutoImageTextAdapter,
    CandidateRuntimeError,
    UHRBATAdapter,
    resolve_uhr_vision_tower,
)


def test_uhr_bat_candidate_is_registered() -> None:
    candidate = CANDIDATES["uhr_bat"]

    assert candidate.loader == "uhr_bat"
    assert candidate.release_verified is True
    assert candidate.license == "Apache-2.0"
    assert ADAPTERS["uhr_bat"] is UHRBATAdapter


def test_uhr_bat_resolves_complete_local_vision_tower(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "models"
        / "openai--clip-vit-large-patch14-336"
        / "snapshots"
        / "main"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "preprocessor_config.json").write_text("{}")
    (snapshot / "pytorch_model.bin").write_bytes(b"weights")

    assert resolve_uhr_vision_tower(tmp_path) == snapshot


def test_uhr_bat_rejects_incomplete_vision_tower(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="clip-vit-large-patch14-336"):
        resolve_uhr_vision_tower(tmp_path)


def test_uhr_bat_chatml_prompt_contains_image_and_question() -> None:
    prompt = UHRBATAdapter._chatml_prompt("How many ships are visible?")

    assert "<image>" in prompt
    assert "How many ships are visible?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_uhr_bat_rejects_multi_image_messages_explicitly() -> None:
    adapter = UHRBATAdapter(CANDIDATES["uhr_bat"], Path("/unused"))
    image = Image.new("RGB", (2, 2))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the changes."},
            ],
        }
    ]

    with pytest.raises(CandidateRuntimeError, match="exactly one image"):
        adapter.generate_messages(messages)

    assert adapter.unsupported == {"levir_cc"}


def test_uhr_bat_neutralizes_repetition_penalty_for_negative_image_token() -> None:
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        @staticmethod
        def decode(token_ids, skip_special_tokens: bool = True) -> str:
            return "Yellow"

    class FakeModel:
        config = SimpleNamespace(image_token_index=-200)
        device = torch.device("cpu")

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                sequences=torch.tensor([[-200, 1, 5]]),
                scores=[torch.zeros(1, 8)],
            )

    model = FakeModel()
    adapter = UHRBATAdapter(CANDIDATES["uhr_bat"], Path("/unused"))
    adapter.model = model
    adapter.tokenizer = FakeTokenizer()
    adapter.processor = object()
    adapter.uhrbat = SimpleNamespace(
        tokenizer_image_token=lambda *args, **kwargs: torch.tensor([-200, 1]),
        split_image_to_multiscale_tiles=lambda *args, **kwargs: torch.zeros(1),
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (2, 2))},
                {"type": "text", "text": "What color is the vehicle?"},
            ],
        }
    ]

    assert adapter.generate_messages(messages) == "Yellow"
    assert model.kwargs["repetition_penalty"] == 1.0


def test_miril_candidate_uses_generic_image_text_adapter() -> None:
    candidate = CANDIDATES["miril_drone_2b1"]

    assert candidate.loader == "auto_image_text"
    assert candidate.release_verified is True
    assert candidate.license == "Apache-2.0"
    assert ADAPTERS["auto_image_text"] is AutoImageTextAdapter


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"answer": "three ships"}', "three ships"),
        ('{"caption": "a harbor"}', "a harbor"),
        ("plain answer", "plain answer"),
    ],
)
def test_auto_image_text_unwraps_documented_json_contract(
    raw: str, expected: str
) -> None:
    assert AutoImageTextAdapter._unwrap_structured_text(raw) == expected
