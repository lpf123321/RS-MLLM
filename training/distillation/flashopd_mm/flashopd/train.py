"""FlashOPD 训练入口：单文件完成从模型加载到训练完成的全流程.

CleanRL 哲学：所有逻辑可见、可 copy-paste、可 hack。
"""
from __future__ import annotations

import json
import os

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
)

from flashopd.config import OPDConfig
from flashopd.teacher import create_teacher
from flashopd.trainer import OPDTrainer
from flashopd.multimodal import (
    MultimodalDataCollator,
    load_multimodal_dataset,
)


def _apply_lora(model, cfg: OPDConfig):
    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in cfg.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_prompt(instruction: str, input_text: str = "") -> str:
    """将 instruction + input 组装成 prompt 文本."""
    if input_text:
        return f"{instruction}\n{input_text}"
    return instruction


def _load_json_or_jsonl(path: str) -> Dataset:
    """Auto-detect JSON (array) vs JSONL (line-delimited) and load safely."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        first_char = f.read(1).strip()

    if first_char == "[":
        data = json.load(open(path, encoding="utf-8", errors="ignore"))
        return Dataset.from_list(data)

    def _gen():
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    return Dataset.from_generator(_gen)


def _get_cache_path(cfg: OPDConfig) -> str | None:
    """Generate a deterministic cache path based on data file and config."""
    if not cfg.cache_dir:
        return None
    import hashlib
    key = f"{cfg.data_path}|{cfg.max_seq_length}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    name = os.path.basename(cfg.data_path).rsplit(".", 1)[0]
    return os.path.join(cfg.cache_dir, f"{name}_{h}")


def prepare_dataset(cfg: OPDConfig, tokenizer):
    """加载并 tokenize 数据.

    支持两种数据格式：
      1. SFT 格式: {"instruction": "...", "input": "...", "output": "..."}
      2. 纯文本: {"text": "..."}（不区分 prompt/response，仅用于纯 CE 训练）

    SFT 格式会生成 labels，prompt 部分标记为 -100（不计算 CE loss），
    同时记录 prompt_length 供 OPD rollout 使用。

    DDP 模式下只有 rank 0 处理数据并缓存，其他 rank 等待后从缓存加载。
    使用文件标记(.done)同步，不依赖 dist.barrier()（此时 Trainer 尚未初始化）。
    """
    import time

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    cache_path = _get_cache_path(cfg)
    if not cache_path and world_size > 1:
        import hashlib
        key = f"{cfg.data_path}|{cfg.max_seq_length}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        name = os.path.basename(cfg.data_path).rsplit(".", 1)[0]
        cache_path = os.path.join(cfg.output_dir, f".cache_{name}_{h}")

    done_marker = f"{cache_path}.done" if cache_path else None

    if cache_path and os.path.isdir(cache_path) and (
        world_size == 1 or (done_marker and os.path.exists(done_marker))
    ):
        if rank == 0:
            print(f"  [FlashOPD] Loading cached dataset from {cache_path}")
        ds = Dataset.load_from_disk(cache_path)
        ds.set_format("torch")
        return ds

    if world_size > 1 and rank != 0:
        print(f"  [Rank {rank}] Waiting for rank 0 to prepare dataset...")
        while not (done_marker and os.path.exists(done_marker)):
            time.sleep(2)
        print(f"  [Rank {rank}] Loading cached dataset from {cache_path}")
        ds = Dataset.load_from_disk(cache_path)
        ds.set_format("torch")
        return ds

    # Only rank 0 (or single-process) reaches here
    if cfg.data_path.endswith((".jsonl", ".json")):
        ds = _load_json_or_jsonl(cfg.data_path)
    else:
        ds = load_dataset(cfg.data_path, split="train")

    is_sft = "instruction" in ds.column_names and "output" in ds.column_names
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    IGNORE_INDEX = -100

    def tokenize_sft(example):
        prompt = build_prompt(example["instruction"], example.get("input", ""))
        response = example["output"]

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + response_ids
        prompt_len = len(prompt_ids)

        if len(full_ids) > cfg.max_seq_length:
            full_ids = full_ids[: cfg.max_seq_length]
            prompt_len = min(prompt_len, cfg.max_seq_length)

        labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
        attn_mask = [1] * len(full_ids)

        pad_len = cfg.max_seq_length - len(full_ids)
        if pad_len > 0:
            full_ids = full_ids + [pad_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len
            attn_mask = attn_mask + [0] * pad_len

        return {
            "input_ids": full_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "prompt_length": prompt_len,
        }

    def tokenize_text(example):
        text_col = "text" if "text" in example else list(example.keys())[0]
        enc = tokenizer(
            example[text_col],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        enc["prompt_length"] = 0
        return enc

    tokenize_fn = tokenize_sft if is_sft else tokenize_text
    cpu_count = os.cpu_count() or 1
    max_workers = 8 if world_size > 1 else 32
    num_workers = min(cpu_count, max_workers)
    if rank == 0:
        print(f"  [FlashOPD] Tokenizing with num_proc={num_workers} ...")
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, num_proc=num_workers)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        ds.save_to_disk(cache_path)
        with open(done_marker, "w") as f:
            f.write("done")
        print(f"  [FlashOPD] Dataset cached to {cache_path}")

    ds.set_format("torch")
    return ds


def run_training(cfg: OPDConfig):
    """从 OPDConfig 启动完整训练流程."""
    rank = int(os.getenv("RANK", "0"))

    if rank == 0:
        print(f"\n  FlashOPD v0.1.0")
        print(f"  Student: {cfg.student_model}")
        print(f"  Teacher: {cfg.teacher_model or cfg.teacher_api_url or 'None'}")
        print(f"  KL type: {cfg.kl_type} | T={cfg.temperature}")
        print(f"  CE={cfg.ce_coef} KL={cfg.kl_coef} balance={cfg.loss_balance}\n")

    # ---- 1. Processor（同时处理文本和图像）----
    processor_kwargs = {"trust_remote_code": True}
    if cfg.min_pixels is not None:
        processor_kwargs["min_pixels"] = cfg.min_pixels
    if cfg.max_pixels is not None:
        processor_kwargs["max_pixels"] = cfg.max_pixels
    processor = AutoProcessor.from_pretrained(cfg.student_model, **processor_kwargs)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---- 2. Student Model ----
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    student = AutoModelForImageTextToText.from_pretrained(
        cfg.student_model,
        dtype=dtype,
        trust_remote_code=True,
    )

    if cfg.use_lora:
        student = _apply_lora(student, cfg)
        if cfg.gradient_checkpointing:
            student.gradient_checkpointing_enable()

    # ---- 3. Teacher ----
    teacher = None
    if cfg.teacher_model or cfg.teacher_api_url:
        teacher = create_teacher(cfg, student_tokenizer=tokenizer)

    # ---- 4. Data ----
    if not cfg.multimodal:
        raise ValueError("flash-opd-mm 仅支持 multimodal: true")
    if cfg.teacher_backend == "api":
        raise NotImplementedError(
            "多模态 API teacher 需要传图像而非纯 token IDs；当前请使用 local teacher"
        )
    full_dataset = load_multimodal_dataset(cfg.data_path, cfg.max_samples)
    collator = MultimodalDataCollator(
        processor=processor,
        image_root=cfg.image_root,
        max_length=cfg.max_seq_length,
        enable_thinking=cfg.enable_thinking,
    )
    eval_dataset = None
    if cfg.eval_data_path:
        dataset = full_dataset
        eval_dataset = load_multimodal_dataset(cfg.eval_data_path, cfg.max_samples)
    elif cfg.eval_strategy != "no" and cfg.eval_split_ratio > 0:
        from torch.utils.data import random_split
        eval_size = max(1, int(len(full_dataset) * cfg.eval_split_ratio))
        train_size = len(full_dataset) - eval_size
        dataset, eval_dataset = random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42),
        )
        if rank == 0:
            print(
                f"  Auto-split: {len(dataset)} train / "
                f"{len(eval_dataset)} eval "
                f"({cfg.eval_split_ratio:.0%})"
            )
    else:
        dataset = full_dataset

    # ---- 5. Training Args ----
    eval_steps = cfg.eval_steps if cfg.eval_steps is not None else cfg.logging_steps
    training_args_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        deepspeed=cfg.deepspeed,
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    if eval_dataset is not None:
        training_args_kwargs["eval_strategy"] = cfg.eval_strategy
        training_args_kwargs["eval_steps"] = eval_steps

    training_args = TrainingArguments(**training_args_kwargs)

    # ---- 6. Trainer ----
    trainer = OPDTrainer(
        opd_config=cfg,
        teacher=teacher,
        model=student,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collator,
    )

    # ---- 7. Train ----
    from transformers.trainer_utils import get_last_checkpoint

    resume_checkpoint = (
        get_last_checkpoint(cfg.output_dir)
        if os.path.isdir(cfg.output_dir)
        else None
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model()

    if rank == 0:
        print("\n  FlashOPD training complete!\n")
