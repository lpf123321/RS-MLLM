"""Model identity denial and trusted-experiment provenance checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ForbiddenModelError(ValueError):
    """Raised before a known untrusted model can be used."""


class UntrustedExperimentModelError(ValueError):
    """Raised when a model does not match a pinned trusted profile."""


FORBIDDEN_MODEL_IDENTIFIERS = frozenset({"qwen3remote", "kaiz0603"})
FORBIDDEN_MODEL_REVISIONS = frozenset(
    {
        "8c43aa83dab6fc3a46eb52575bc56f8f3122b808",
        "813c7ec6fd3f8a456fb6a42fa6d4e79de96a7d4d",
    }
)
FORBIDDEN_MODEL_LAYOUTS: tuple[dict[str, int], ...] = (
    {
        "config.json": 1_942,
        "model-00001-of-00002.safetensors": 4_967_229_296,
        "model-00002-of-00002.safetensors": 3_908_490_048,
        "model.safetensors.index.json": 64_742,
    },
    {"config.json": 1_914, "model.safetensors": 4_255_140_312},
)

# Suffixes of loader-recognized weight artifacts (same convention as the
# response-level distillation producer's runtime_artifact_policy). A trusted
# profile fails when a directory contains any recognized weight artifact the
# profile does not pin, so a stale monolithic file, stray shard, pytorch bin,
# adapter weight or GGUF next to the expected layout is never silently loaded.
_WEIGHT_FILE_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".safetensors.index.json",
    ".bin",
    ".bin.index.json",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
)

# Runtime support files selected by the transformers/vLLM loaders. Derived
# candidate profiles must list every present file of this kind, so a
# loader-selected tokenizer/processor/template/config artifact can never
# silently escape the pinned set.
LOADER_RUNTIME_FILE_NAMES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "image_processor_config.json",
        "audio_processor_config.json",
    }
)


def _is_weight_name(name: str) -> bool:
    return name.endswith(_WEIGHT_FILE_SUFFIXES)


def recognized_weight_files(root: Path) -> set[str]:
    """Names of loader-recognized weight artifacts present under ``root``."""
    recognized: set[str] = set()
    try:
        entries = list(root.iterdir())
    except OSError:
        return recognized
    for artifact in entries:
        if not artifact.is_file():
            continue
        name = artifact.name
        if _is_weight_name(name):
            recognized.add(name)
    return recognized


@dataclass(frozen=True)
class TrustedModelProfile:
    key: str
    model_ids: frozenset[str]
    revision: str
    source_url: str
    license_id: str
    license_url: str
    required_files: dict[str, int]
    sha256: dict[str, str]
    runtime_files: dict[str, int] = field(default_factory=dict)
    runtime_sha256: dict[str, str] = field(default_factory=dict)
    baseline_exception: bool = False


TRUSTED_MODEL_PROFILES: dict[str, TrustedModelProfile] = {
    "qwen35_4b": TrustedModelProfile(
        key="qwen35_4b",
        model_ids=frozenset({"qwen/qwen3.5-4b", "qwen35_4b"}),
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        source_url="https://huggingface.co/Qwen/Qwen3.5-4B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        required_files={
            "config.json": 3_161,
            "model.safetensors-00001-of-00002.safetensors": 5_329_398_688,
            "model.safetensors-00002-of-00002.safetensors": 3_990_429_408,
            "model.safetensors.index.json": 76_196,
        },
        sha256={
            "config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
            "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
            "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
            "model.safetensors.index.json": "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
        },
        # Runtime tokenizer/processor/chat-template/config artifacts pinned
        # from the source-verified local snapshot
        # (ModelScope Qwen/Qwen3.5-4B@master, /home/u2024311009/RS-MLLM/.ms_cache/
        # models/Qwen--Qwen3.5-4B/snapshots/master, sha256-verified 2026-08-20).
        runtime_files={
            "chat_template.jinja": 7_756,
            "merges.txt": 3_353_259,
            "preprocessor_config.json": 390,
            "tokenizer_config.json": 16_710,
            "tokenizer.json": 12_807_982,
            "video_preprocessor_config.json": 385,
            "vocab.json": 6_722_759,
        },
        runtime_sha256={
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
            "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        },
    ),
    "qwen35_4b_awq": TrustedModelProfile(
        key="qwen35_4b_awq",
        model_ids=frozenset({"cyankiwi/qwen3.5-4b-awq-4bit", "qwen35_4b_awq"}),
        revision="ef85d23bebaba87b3c4672ba11c449c79dbdb23e",
        source_url="https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        required_files={
            "config.json": 11_865,
            "model-00001-of-00001.safetensors": 4_040_461_440,
            "model.safetensors.index.json": 111_479,
        },
        sha256={
            "config.json": "6ab1e4f1d1695f687accde679fee33835b848b1825f1948a3d558c1a399bb7dc",
            "model-00001-of-00001.safetensors": "902477edf53bc6768bd1f212dd1866856fd5a0627def06887780c95900ffb013",
            "model.safetensors.index.json": "bb34e4b964f43d96d97866f7f438019315359784721a44886d7d2e7737317114",
        },
        # Runtime tokenizer/processor/chat-template/config from the quantized
        # repo (identical hashes to the Qwen3.5-4B base snapshot, i.e. same
        # tokenizer/processor lineage), sha256-verified 2026-08-26.
        runtime_files={
            "chat_template.jinja": 7_756,
            "generation_config.json": 120,
            "merges.txt": 3_353_259,
            "preprocessor_config.json": 390,
            "tokenizer_config.json": 16_710,
            "tokenizer.json": 12_807_982,
            "video_preprocessor_config.json": 385,
            "vocab.json": 6_722_759,
        },
        runtime_sha256={
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "generation_config.json": "a33e6e9d6ec02519266a2b1d6eb573cb2bd83e46a021a6d59ab605f86aecaf3a",
            "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
            "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        },
    ),
    "mmerestore_bf16": TrustedModelProfile(
        key="mmerestore_bf16",
        model_ids=frozenset({"mmerestore_bf16"}),
        revision="local-snapshot",
        source_url="https://huggingface.co/Qwen/Qwen3.5-4B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        required_files={
            "model-00001-of-00002.safetensors": 4972947968,
            "model-00002-of-00002.safetensors": 4105672464,
            "model.safetensors.index.json": 66236,
        },
        sha256={
            "model-00001-of-00002.safetensors": "fb0fc3442a46e4f94b3f3243318cd1cb403fa967a9c605202063ea39fc28b4ae",
            "model-00002-of-00002.safetensors": "ec7ebdaa452b8859b19dc4754d04dcc31d4750ddd651daa0ee65935e2bcee212",
            "model.safetensors.index.json": "df59e0e8f68ce1841a11ec0ba05e468a1413a7c5df5d1fa1e8b29a911a1b29d1",
        },
        # runtime files (tokenizer/processor/config)
        runtime_files={
            "args.json": 17655,
            "chat_template.jinja": 7756,
            "config.json": 2854,
            "config.json.bak.2026-08-20-035031": 2946,
            "generation_config.json": 138,
            "preprocessor_config.json": 390,
            "processor_config.json": 1191,
            "tokenizer.json": 19989325,
            "tokenizer_config.json": 1165,
        },
        runtime_sha256={
            "args.json": "282031a1df896c32c9e642f713a801853abb323261a5231aabaf5e9eca46ef20",
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "config.json": "d6b4ebb8f9dedae345917774f2fe9560a845d3f692930ca4b7019b895577dcf7",
            "config.json.bak.2026-08-20-035031": "1dfda2230c188c16294924e6d4211f311e9d9dfe4a0e59e9e242916ff015d8cc",
            "generation_config.json": "f7c35afc4eda09ee1a9cbe029b030dc9875ab3c201401c48044448e8153acea2",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
        },
    ),
    "mmerestore_w8a8": TrustedModelProfile(
        key="mmerestore_w8a8",
        model_ids=frozenset({"mmerestore_w8a8"}),
        revision="local-snapshot",
        source_url="https://huggingface.co/Qwen/Qwen3.5-4B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        required_files={
            "model-00001-of-00003.safetensors": 1980847192,
            "model-00002-of-00003.safetensors": 1983228104,
            "model-00003-of-00003.safetensors": 1551671800,
            "model.safetensors.index.json": 87006,
        },
        sha256={
            "model-00001-of-00003.safetensors": "a13ed14ba1c038d0ffaa354035f15aad70f95bcec56396f7c0cdcfecda03f876",
            "model-00002-of-00003.safetensors": "302b7c5e961aad1fb79938c3c898002ec08cbe7e94afdeaf008ee63fb61baa95",
            "model-00003-of-00003.safetensors": "b5d07b053b720aa43daf5b417e93b2a1c58e0453bbcb5e442398c1c8c8bb3f77",
            "model.safetensors.index.json": "cac5fb210daad2bbaa68a8c546f4d34e2b25eb2a3535ef117ed112665cfd1afb",
        },
        # runtime files (tokenizer/processor/config)
        runtime_files={
            "chat_template.jinja": 7756,
            "config.json": 14004,
            "conversion_manifest.json": 4172,
            "generation_config.json": 138,
            "processor_config.json": 1191,
            "recipe.yaml": 895,
            "tokenizer.json": 19989424,
            "tokenizer_config.json": 1213,
        },
        runtime_sha256={
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "config.json": "7a3f70a90352e2d19ec059067f96e49ab0b053850258f246b4d13f29130d4d51",
            "conversion_manifest.json": "915e88664d52df102c5759c9ee9ef237425b24ed5ce1a91faa7d969103787d7b",
            "generation_config.json": "6b4c2f2ec530620dd17494f351841d8212ea8ba36fe1afccd19fe0a29a70b34c",
            "processor_config.json": "7b1135e81a95bc631c697a964aee3e36fcc9529aac996b81a8efd105fbb6c3b4",
            "recipe.yaml": "fc6b4a371efd0f9143a7b0bbc72271fc9beaf88187122ea7b9e82c645c4d62c4",
            "tokenizer.json": "d73c2c5f7aa0ed522c8d96ef3524739eb61e3c78e74839a2ce4a1c56ea340a20",
            "tokenizer_config.json": "408a2f228cac922074851da3ce458f8eb339f2d6e1a3e11c358822bd0aac0f5c",
        },
    ),
    "mmerestore_gptq": TrustedModelProfile(
        key="mmerestore_gptq",
        model_ids=frozenset({"mmerestore_gptq"}),
        revision="local-snapshot",
        source_url="https://huggingface.co/Qwen/Qwen3.5-4B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        required_files={
            "model-00001-of-00002.safetensors": 1997220608,
            "model-00002-of-00002.safetensors": 1789422712,
            "model.safetensors.index.json": 109176,
        },
        sha256={
            "model-00001-of-00002.safetensors": "7244355144427406d785fc539b770e1604ad035d76b9fb2fdb5ac37029dd9a6f",
            "model-00002-of-00002.safetensors": "093499db3f20bfd6488863936d18d49bdc71c9531c2242ed60a61bdd18bcfcdc",
            "model.safetensors.index.json": "77734ad5291e1a0c779ca2137f82b26e6381482fd8aadd25d5532b1620f25f18",
        },
        # runtime files (tokenizer/processor/config)
        runtime_files={
            "chat_template.jinja": 7756,
            "config.json": 13647,
            "conversion_manifest.json": 3371,
            "generation_config.json": 138,
            "processor_config.json": 1191,
            "recipe.yaml": 440,
            "tokenizer.json": 19989424,
            "tokenizer_config.json": 1213,
        },
        runtime_sha256={
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "config.json": "962b005f3988554a444125f535e21e3d4539f9a44f5b2c5c32fd1d19cd3ddf5f",
            "conversion_manifest.json": "497592c992c34d6fdfe5b1d173e08b072eee55f8b7453a2407cd9d749080a129",
            "generation_config.json": "6b4c2f2ec530620dd17494f351841d8212ea8ba36fe1afccd19fe0a29a70b34c",
            "processor_config.json": "7b1135e81a95bc631c697a964aee3e36fcc9529aac996b81a8efd105fbb6c3b4",
            "recipe.yaml": "1795c18e19c873e59eb992dca0c8642055cb4ceacf5609d8f1517702cf7336c8",
            "tokenizer.json": "d73c2c5f7aa0ed522c8d96ef3524739eb61e3c78e74839a2ce4a1c56ea340a20",
            "tokenizer_config.json": "408a2f228cac922074851da3ce458f8eb339f2d6e1a3e11c358822bd0aac0f5c",
        },
    ),
    "pruned_w20": TrustedModelProfile(
        key="pruned_w20",
        model_ids=frozenset({"pruned_w20", "mmerestore_pruned_w20"}),
        revision="local-snapshot",
        source_url="https://huggingface.co/Qwen/Qwen3.5-4B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE",
        required_files={
            "model-00001-of-00002.safetensors": 4972947968,
            "model-00002-of-00002.safetensors": 4105672432,
            "model.safetensors.index.json": 66236,
        },
        sha256={
            "model-00001-of-00002.safetensors": "fb0fc3442a46e4f94b3f3243318cd1cb403fa967a9c605202063ea39fc28b4ae",
            "model-00002-of-00002.safetensors": "1bf0189dd6ce186a3b02eb2b48ea7597a5c0a28790fcac9bfe8d113721e64b8e",
            "model.safetensors.index.json": "df59e0e8f68ce1841a11ec0ba05e468a1413a7c5df5d1fa1e8b29a911a1b29d1",
        },
        runtime_files={
            "chat_template.jinja": 7756,
            "config.json": 2854,
            "generation_config.json": 138,
            "processor_config.json": 1191,
            "preprocessor_config.json": 390,
            "tokenizer.json": 19989325,
            "tokenizer_config.json": 1165,
        },
        runtime_sha256={
            "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            "config.json": "d6b4ebb8f9dedae345917774f2fe9560a845d3f692930ca4b7019b895577dcf7",
            "generation_config.json": "f7c35afc4eda09ee1a9cbe029b030dc9875ab3c201401c48044448e8153acea2",
            "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
        },
    ),
    "qwen35_2b": TrustedModelProfile(
        key="qwen35_2b",
        model_ids=frozenset({"qwen/qwen3.5-2b", "qwen35_2b"}),
        revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        source_url="https://huggingface.co/Qwen/Qwen3.5-2B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE",
        required_files={
            "config.json": 2_908,
            "model.safetensors-00001-of-00001.safetensors": 4_548_221_488,
            "model.safetensors.index.json": 64_460,
        },
        sha256={
            "config.json": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
            "model.safetensors-00001-of-00001.safetensors": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
            "model.safetensors.index.json": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
        },
        # Runtime tokenizer/processor/chat-template/config artifacts pinned
        # from the source-verified local snapshot
        # (ModelScope Qwen/Qwen3.5-2B@master, /home/u2024311009/RS-MLLM/.ms_cache/
        # models/Qwen--Qwen3.5-2B/snapshots/master, sha256-verified 2026-08-20;
        # weight hashes cross-checked against exec-compression/qwen35_2b_source.json).
        runtime_files={
            "chat_template.jinja": 7_755,
            "merges.txt": 3_353_259,
            "preprocessor_config.json": 390,
            "tokenizer_config.json": 16_709,
            "tokenizer.json": 12_807_982,
            "video_preprocessor_config.json": 385,
            "vocab.json": 6_722_759,
        },
        runtime_sha256={
            "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
            "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
            "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
            "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        },
    ),
    "qwen3vl_2b_baseline": TrustedModelProfile(
        key="qwen3vl_2b_baseline",
        model_ids=frozenset({"qwen/qwen3-vl-2b-instruct", "qwen3vl_2b_baseline"}),
        revision="89644892e4d85e24eaac8bacfd4f463576704203",
        source_url="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
        required_files={"config.json": 1_505, "model.safetensors": 4_255_140_312},
        sha256={
            "config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
            "model.safetensors": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
        },
        baseline_exception=True,
    ),
}

DERIVED_PROFILE_SCHEMA_VERSION = 1
DERIVED_PROFILE_KIND = "derived_candidate_profile"


@dataclass(frozen=True)
class DerivedModelProfile:
    """External profile for a locally merged/trained candidate model.

    A candidate (e.g. a LoRA-merged Qwen3.5 checkpoint) is evaluated only
    through one of these manifests: exact model path, trusted base profile
    and revision, license linkage, training/fused evidence, and every
    loader-selected weight and runtime support file with exact size + SHA-256.
    """

    key: str
    model_path: Path
    base_profile_key: str
    base_revision: str
    license_id: str
    license_url: str
    source_url: str
    evidence: dict[str, Any]
    weights: dict[str, int]
    weights_sha256: dict[str, str]
    runtime_files: dict[str, int]
    runtime_sha256: dict[str, str]

    def as_trusted_profile(self) -> TrustedModelProfile:
        return TrustedModelProfile(
            key=self.key,
            model_ids=frozenset({self.key}),
            revision=self.base_revision,
            source_url=self.source_url,
            license_id=self.license_id,
            license_url=self.license_url,
            required_files=dict(self.weights),
            sha256=dict(self.weights_sha256),
            runtime_files=dict(self.runtime_files),
            runtime_sha256=dict(self.runtime_sha256),
        )


def _matching_forbidden_tokens(value: str) -> list[str]:
    normalized = value.casefold()
    tokens = FORBIDDEN_MODEL_IDENTIFIERS | FORBIDDEN_MODEL_REVISIONS
    return sorted(token for token in tokens if token in normalized)


def _candidate_roots(path: Path) -> tuple[Path, ...]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return (resolved.parent,)
    roots = [resolved]
    snapshots = resolved / "snapshots"
    if snapshots.is_dir():
        roots.extend(entry for entry in snapshots.iterdir() if entry.is_dir())
    return tuple(roots)


def _matches_layout(root: Path, files: dict[str, int]) -> bool:
    try:
        return all(
            (root / name).is_file() and (root / name).stat().st_size == size
            for name, size in files.items()
        )
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_weight_files(profile: TrustedModelProfile) -> set[str]:
    return {name for name in profile.required_files if _is_weight_name(name)}


def _unexpected_weight_files(root: Path, profile: TrustedModelProfile) -> list[str]:
    """Recognized weight artifacts under ``root`` that the profile does not pin."""
    expected = _profile_weight_files(profile)
    return sorted(recognized_weight_files(root) - expected)


def _profile_violations(root: Path, profile: TrustedModelProfile) -> list[str]:
    """Fail-closed layout violations for a profile: unexpected recognized
    weights and, for profiles that pin runtime artifacts, loader-selected
    runtime files absent from the pinned set (a stale processor/template/
    tokenizer file must not silently override pinned assets)."""
    violations: list[str] = []
    unexpected_weights = _unexpected_weight_files(root, profile)
    if unexpected_weights:
        violations.append(
            "unexpected recognized weight artifacts: " + ", ".join(unexpected_weights)
        )
    if profile.runtime_files:
        try:
            present = {entry.name for entry in root.iterdir() if entry.is_file()}
        except OSError:
            present = set()
        pinned = set(profile.required_files) | set(profile.runtime_files)
        unlisted = sorted((LOADER_RUNTIME_FILE_NAMES & present) - pinned)
        if unlisted:
            violations.append(
                "unlisted loader-selected runtime files: " + ", ".join(unlisted)
            )
    return violations


def _matches_profile(root: Path, profile: TrustedModelProfile) -> bool:
    if not _matches_layout(root, profile.required_files):
        return False
    if profile.runtime_files and not _matches_layout(root, profile.runtime_files):
        return False
    if set(profile.runtime_files) != set(profile.runtime_sha256):
        return False
    if _profile_violations(root, profile):
        return False
    expected_hashes = {**profile.sha256, **profile.runtime_sha256}
    try:
        return all(
            _sha256(root / name) == expected
            for name, expected in expected_hashes.items()
        )
    except OSError:
        return False


def _metadata_forbidden_tokens(root: Path) -> list[str]:
    matched: set[str] = set()
    for name in ("config.json", "adapter_config.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        matched.update(_matching_forbidden_tokens(json.dumps(payload, ensure_ascii=False)))
    return sorted(matched)


def assert_model_allowed(identifier: str | Path | None) -> None:
    if identifier is None:
        return
    matched = _matching_forbidden_tokens(str(identifier))
    path = Path(identifier).expanduser()
    if path.exists():
        for root in _candidate_roots(path):
            matched.extend(_metadata_forbidden_tokens(root))
            if any(_matches_layout(root, layout) for layout in FORBIDDEN_MODEL_LAYOUTS):
                matched.append("forbidden artifact layout")
    if matched:
        raise ForbiddenModelError(
            "Untrusted model is permanently disabled because its training provenance "
            "and benchmark contamination status are unknown "
            f"(matched: {', '.join(sorted(set(matched)))})."
        )


def trusted_profile_for_request(identifier: str) -> TrustedModelProfile:
    assert_model_allowed(identifier)
    normalized = identifier.strip().casefold()
    for profile in TRUSTED_MODEL_PROFILES.values():
        if normalized in profile.model_ids:
            return profile
    raise UntrustedExperimentModelError(f"{identifier!r} is not in the trusted experiment allowlist.")


def verify_trusted_model_content(path: str | Path, profile_key: str | None = None) -> TrustedModelProfile:
    assert_model_allowed(path)
    model_path = Path(path).expanduser()
    if not model_path.exists():
        raise UntrustedExperimentModelError(f"Model path does not exist: {model_path}")
    profiles = ((TRUSTED_MODEL_PROFILES[profile_key],) if profile_key is not None else tuple(TRUSTED_MODEL_PROFILES.values()))
    for root in _candidate_roots(model_path):
        for profile in profiles:
            if _matches_profile(root, profile):
                return profile
    violations: list[str] = []
    for root in _candidate_roots(model_path):
        for profile in profiles:
            violations.extend(
                f"{root}: {detail}" for detail in _profile_violations(root, profile)
            )
    expected = profile_key or "one of: " + ", ".join(TRUSTED_MODEL_PROFILES)
    message = (
        f"Local model content at {model_path} does not match trusted profile {expected}. "
        "A model name or directory basename is not provenance evidence."
    )
    if violations:
        message += " " + "; ".join(sorted(set(violations))) + "."
    raise UntrustedExperimentModelError(message)


def validate_provenance_envelope(provenance: Any) -> None:
    """Fail-closed envelope for derived-candidate training/fused evidence.

    The provenance must declare a complete full-scope run with zero failed
    rows; when teacher summaries are embedded they must be exactly
    ``mmerestore`` and ``vision_opd_9b``, each complete/full with zero
    failures.
    """
    if not isinstance(provenance, dict) or not provenance:
        raise UntrustedExperimentModelError(
            "Derived candidate evidence has no provenance record"
        )
    checked = False
    if provenance.get("status") is not None or provenance.get("scope") is not None:
        if provenance.get("status") != "complete" or provenance.get("scope") != "full":
            raise UntrustedExperimentModelError(
                "Derived candidate provenance must be status=complete scope=full, "
                f"got {provenance.get('status')!r}/{provenance.get('scope')!r}"
            )
        checked = True
    rows_failed = provenance.get("rows_failed")
    if not isinstance(rows_failed, int) or isinstance(rows_failed, bool) or rows_failed != 0:
        raise UntrustedExperimentModelError(
            f"Derived candidate provenance rows_failed must be exactly 0, "
            f"got {rows_failed!r}"
        )
    teachers = provenance.get("teachers")
    if teachers is not None:
        if not isinstance(teachers, dict) or set(teachers) != {"mmerestore", "vision_opd_9b"}:
            raise UntrustedExperimentModelError(
                "Derived candidate provenance teachers must be exactly "
                "mmerestore and vision_opd_9b"
            )
        for name, entry in teachers.items():
            if not isinstance(entry, dict):
                raise UntrustedExperimentModelError(
                    f"Teacher {name!r} provenance record is malformed"
                )
            if entry.get("status") != "complete" or entry.get("scope") != "full":
                raise UntrustedExperimentModelError(
                    f"Teacher {name!r} provenance must be complete/full, "
                    f"got {entry.get('status')!r}/{entry.get('scope')!r}"
                )
            teacher_failed = entry.get("rows_failed")
            if (
                not isinstance(teacher_failed, int)
                or isinstance(teacher_failed, bool)
                or teacher_failed != 0
            ):
                raise UntrustedExperimentModelError(
                    f"Teacher {name!r} provenance rows_failed must be exactly 0, "
                    f"got {teacher_failed!r}"
                )
        checked = True
    if not checked:
        raise UntrustedExperimentModelError(
            "Derived candidate provenance must declare status/scope or a teachers map"
        )


def _parse_pinned_file_map(value: Any, label: str) -> tuple[dict[str, int], dict[str, str]]:
    if not isinstance(value, dict) or not value:
        raise UntrustedExperimentModelError(
            f"Derived candidate profile {label} must be a non-empty object"
        )
    sizes: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for name, entry in value.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name.startswith("/")
        ):
            raise UntrustedExperimentModelError(f"Invalid {label} file name: {name!r}")
        if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
            raise UntrustedExperimentModelError(
                f"{label} entry {name!r} must be {{'bytes': int, 'sha256': hex}}"
            )
        size = entry["bytes"]
        digest = entry["sha256"]
        if not isinstance(size, int) or size <= 0:
            raise UntrustedExperimentModelError(
                f"{label} entry {name!r} has an invalid byte size"
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise UntrustedExperimentModelError(
                f"{label} entry {name!r} has an invalid sha256"
            )
        sizes[name] = size
        hashes[name] = digest
    return sizes, hashes


def load_derived_profile(path: str | Path) -> DerivedModelProfile:
    """Parse and validate an external derived candidate profile manifest.

    Enforces the manifest structure and its linkage to a trusted base
    profile (exact revision and license); the on-disk model directory is
    enforced separately by ``verify_derived_model_content``.
    """
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise UntrustedExperimentModelError(
            f"Derived candidate profile manifest does not exist: {manifest_path}"
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UntrustedExperimentModelError(
            f"Derived candidate profile manifest is not valid JSON: {manifest_path}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise UntrustedExperimentModelError("Derived candidate profile must be a JSON object")
    if data.get("schema_version") != DERIVED_PROFILE_SCHEMA_VERSION:
        raise UntrustedExperimentModelError(
            f"Unsupported derived profile schema_version: {data.get('schema_version')!r}"
        )
    if data.get("kind") != DERIVED_PROFILE_KIND:
        raise UntrustedExperimentModelError(
            f"Derived profile kind must be {DERIVED_PROFILE_KIND!r}, "
            f"got {data.get('kind')!r}"
        )
    key = data.get("key")
    if not isinstance(key, str) or not key or any(char.isspace() for char in key):
        raise UntrustedExperimentModelError(f"Derived profile has no valid key: {key!r}")
    model_path_value = data.get("model_path")
    if not isinstance(model_path_value, str) or not model_path_value:
        raise UntrustedExperimentModelError("Derived profile has no model_path")
    model_path = Path(model_path_value)
    if not model_path.is_absolute():
        raise UntrustedExperimentModelError(
            f"Derived profile model_path must be absolute: {model_path_value!r}"
        )
    base_key = data.get("base_profile")
    base = TRUSTED_MODEL_PROFILES.get(base_key) if isinstance(base_key, str) else None
    if base is None:
        raise UntrustedExperimentModelError(
            f"Derived profile base_profile {base_key!r} is not a trusted profile"
        )
    base_revision = data.get("base_revision")
    if base_revision != base.revision:
        raise UntrustedExperimentModelError(
            f"Derived profile base_revision {base_revision!r} does not match trusted "
            f"base profile {base.key!r} revision {base.revision!r}"
        )
    license_info = data.get("license")
    if (
        not isinstance(license_info, dict)
        or license_info.get("id") != base.license_id
        or not isinstance(license_info.get("url"), str)
        or not license_info["url"]
    ):
        raise UntrustedExperimentModelError(
            f"Derived profile license does not match trusted base profile {base.key!r}"
        )
    evidence = data.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise UntrustedExperimentModelError("Derived profile has no evidence")
    evidence_kind = evidence.get("kind")
    if not isinstance(evidence_kind, str) or not evidence_kind:
        raise UntrustedExperimentModelError("Derived profile evidence has no kind")
    validate_provenance_envelope(evidence.get("provenance"))
    weights, weights_sha256 = _parse_pinned_file_map(data.get("weights"), "weights")
    runtime_files, runtime_sha256 = _parse_pinned_file_map(
        data.get("runtime_files"), "runtime_files"
    )
    if "config.json" not in runtime_files:
        raise UntrustedExperimentModelError(
            "Derived profile runtime_files must include config.json"
        )
    overlap = set(weights) & set(runtime_files)
    if overlap:
        raise UntrustedExperimentModelError(
            "Derived profile files overlap between weights and runtime_files: "
            + ", ".join(sorted(overlap))
        )
    return DerivedModelProfile(
        key=key,
        model_path=model_path,
        base_profile_key=base.key,
        base_revision=base.revision,
        license_id=base.license_id,
        license_url=license_info["url"],
        source_url=base.source_url,
        evidence=dict(evidence),
        weights=weights,
        weights_sha256=weights_sha256,
        runtime_files=runtime_files,
        runtime_sha256=runtime_sha256,
    )


def verify_derived_model_content(
    profile: DerivedModelProfile, *, hash_files: bool = True
) -> dict[str, Any]:
    """Enforce the on-disk derived profile for a candidate model directory.

    Fails unless: the directory exists, every listed weight and runtime file
    is present with the exact pinned size (and, when ``hash_files`` is set,
    the exact pinned SHA-256), the set of loader-recognized weight artifacts
    matches the listed weights exactly (no unexpected recognized weights),
    and every loader-selected runtime file present is listed. ``hash_files``
    may be disabled for cheap pre-flight layout checks; the full hash
    verification must run at least once before inference (the evaluator does
    this in-process).
    """
    model_path = profile.model_path.resolve()
    if not model_path.is_dir():
        raise UntrustedExperimentModelError(
            f"Derived model path is not a directory: {model_path}"
        )
    try:
        present = {entry.name for entry in model_path.iterdir() if entry.is_file()}
    except OSError as error:
        raise UntrustedExperimentModelError(
            f"Cannot list derived model directory {model_path}: {error}"
        ) from error
    actual_weights = recognized_weight_files(model_path)
    listed_weights = set(profile.weights)
    if actual_weights != listed_weights:
        missing = sorted(listed_weights - actual_weights)
        unexpected = sorted(actual_weights - listed_weights)
        raise UntrustedExperimentModelError(
            "Derived model weight set mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    unlisted_runtime = sorted(
        (LOADER_RUNTIME_FILE_NAMES & present) - set(profile.runtime_files)
    )
    if unlisted_runtime:
        raise UntrustedExperimentModelError(
            "Derived model has loader-selected runtime files absent from the profile: "
            + ", ".join(unlisted_runtime)
        )
    pinned: dict[str, tuple[int, str]] = {}
    for name, size in profile.weights.items():
        pinned[name] = (size, profile.weights_sha256[name])
    for name, size in profile.runtime_files.items():
        pinned[name] = (size, profile.runtime_sha256[name])
    verified: dict[str, dict[str, Any]] = {}
    for name, (expected_size, expected_sha256) in pinned.items():
        artifact = model_path / name
        try:
            if not artifact.is_file():
                raise UntrustedExperimentModelError(
                    f"Derived model artifact is missing: {artifact}"
                )
            actual_size = artifact.stat().st_size
            if actual_size != expected_size:
                raise UntrustedExperimentModelError(
                    f"Derived model artifact size mismatch for {artifact}: "
                    f"{actual_size} != {expected_size}"
                )
            actual_sha256 = _sha256(artifact) if hash_files else expected_sha256
        except OSError as error:
            raise UntrustedExperimentModelError(
                f"Cannot read derived model artifact {artifact}: {error}"
            ) from error
        if actual_sha256 != expected_sha256:
            raise UntrustedExperimentModelError(
                f"Derived model artifact hash mismatch for {artifact}"
            )
        verified[name] = {"bytes": actual_size, "sha256": actual_sha256}
    return {
        "model_path": str(model_path),
        "weights": sorted(profile.weights),
        "runtime_files": sorted(profile.runtime_files),
        "verified_files": verified,
    }


def register_derived_profile(profile: DerivedModelProfile) -> TrustedModelProfile:
    """Register a verified derived profile in-process so run_config can pin it."""
    trusted = profile.as_trusted_profile()
    existing = TRUSTED_MODEL_PROFILES.get(trusted.key)
    if existing is not None and existing != trusted:
        raise RuntimeError(
            f"Conflicting trusted model profile for derived key {trusted.key!r}"
        )
    TRUSTED_MODEL_PROFILES[trusted.key] = trusted
    return trusted


def assert_trusted_experiment_model(identifier: str | Path) -> TrustedModelProfile:
    """Require hash-pinned local content; Hub IDs are download-only inputs."""
    path = Path(identifier).expanduser()
    if path.exists():
        return verify_trusted_model_content(path)
    trusted_profile_for_request(str(identifier))
    raise UntrustedExperimentModelError(
        "Experiment loaders require a local SHA-256-pinned model path; "
        "download the approved revision before use."
    )
