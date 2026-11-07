"""Activation extraction.

One forward pass per batch with ``output_hidden_states=True``. For every prompt
the pass yields two things, both read at the last *real* token:

* the residual-stream vector at every layer, as a float32 array of shape
  ``[n_layers + 1, n_prompts, d]`` (index 0 is the embedding output);
* the model's own *refusal log-odds* for the next token: log-probability mass on
  refusal openers ("I'm sorry", "I cannot", ...) minus mass on compliance openers
  ("Sure", "Here", ...). With a chat template the last token is the end of the
  assistant cue, so this is the first answer token the model would produce.

A steering vector can be added to the residual stream at one layer during the
pass, which is how the scanner tests whether a latent direction *drives* refusal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

REFUSAL_PHRASES = ["I'm sorry", "I am sorry", "Sorry", "I cannot", "I can't", "I apologize",
                   "As an AI", "Unfortunately"]
COMPLIANCE_PHRASES = ["Sure", "Here", "Certainly", "Of course", "Yes", "To", "The", "You",
                      "Subject", "1"]

Steer = tuple[int, np.ndarray]  # (hidden-state index, vector added at that layer's output)


class Extractor(Protocol):
    n_layers: int
    hidden_size: int

    def extract(self, prompts: list[str], batch_size: int = 8) -> np.ndarray: ...

    def extract_with_refusal(self, prompts: list[str], batch_size: int = 8,
                             steer: Optional[Steer] = None) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class LoadOptions:
    device: str = "auto"  # auto | cpu | cuda | mps
    dtype: str = "auto"  # auto | float32 | float16 | bfloat16
    trust_remote_code: bool = False
    allow_pickle: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    chat_template: bool = True  # wrap prompts in the model's chat template if it has one
    max_length: int = 256


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float32 if device == "cpu" else torch.bfloat16
    return getattr(torch, dtype)


def first_token_ids(tokenizer, phrases: list[str]) -> list[int]:
    """Distinct first-token ids of the given phrases (how a reply would start)."""
    ids = []
    for p in phrases:
        enc = tokenizer(p, add_special_tokens=False)["input_ids"]
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def decoder_layers(model):
    """The list of transformer blocks, for the common HF layouts."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise RuntimeError(f"Cannot find decoder layers on {type(model).__name__}; steering unsupported")


class HFExtractor:
    """Extractor backed by a plain HuggingFace causal LM."""

    def __init__(self, model, tokenizer, device: str, use_chat_template: bool = True,
                 max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.n_layers = int(model.config.num_hidden_layers)
        self.hidden_size = int(model.config.hidden_size)
        self.use_chat_template = bool(use_chat_template and tokenizer.chat_template)
        if use_chat_template and not tokenizer.chat_template:
            logger.warning("Tokenizer has no chat template; using raw prompts")
        # Left padding + mask-based gather makes the last-token read independent of batch size.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        ref = first_token_ids(tokenizer, REFUSAL_PHRASES)
        com = first_token_ids(tokenizer, COMPLIANCE_PHRASES)
        self.refusal_ids = [i for i in ref if i not in com]
        self.comply_ids = [i for i in com if i not in ref]
        if not self.refusal_ids or not self.comply_ids:
            logger.warning("Tokenizer cannot separate refusal from compliance openers; "
                           "refusal log-odds will be 0")

    @classmethod
    def from_pretrained(cls, model_path: str, opts: LoadOptions | None = None) -> "HFExtractor":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        opts = opts or LoadOptions()
        device = resolve_device(opts.device)
        dtype = resolve_dtype(opts.dtype, device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=opts.trust_remote_code
        )
        kwargs: dict = {"trust_remote_code": opts.trust_remote_code}
        if not opts.allow_pickle:
            kwargs["use_safetensors"] = True
        if opts.load_in_8bit or opts.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=opts.load_in_8bit, load_in_4bit=opts.load_in_4bit
            )
            kwargs["device_map"] = device
        else:
            kwargs["dtype"] = dtype

        try:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            except TypeError:  # transformers < 4.56 spells it torch_dtype
                kwargs["torch_dtype"] = kwargs.pop("dtype")
                model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        except OSError as e:
            if not opts.allow_pickle and "safetensors" in str(e).lower():
                raise RuntimeError(
                    f"No safetensors weights for '{model_path}'. Pickle (.bin) checkpoints can "
                    "run arbitrary code on load; pass allow_pickle=True only for trusted models."
                ) from e
            raise
        if not (opts.load_in_8bit or opts.load_in_4bit):
            model.to(device)
        logger.info("Loaded %s: %d layers, d=%d, device=%s, dtype=%s",
                    type(model).__name__, model.config.num_hidden_layers,
                    model.config.hidden_size, device, dtype)
        return cls(model, tokenizer, device, opts.chat_template, opts.max_length)

    def format(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def extract(self, prompts: list[str], batch_size: int = 8) -> np.ndarray:
        return self.extract_with_refusal(prompts, batch_size)[0]

    def _refusal_logodds(self, logits: torch.Tensor) -> torch.Tensor:
        """log P(refusal openers) - log P(compliance openers) from next-token logits [B, V]."""
        if not self.refusal_ids or not self.comply_ids:
            return torch.zeros(logits.shape[0], device=logits.device)
        lp = logits.float().log_softmax(-1)
        return torch.logsumexp(lp[:, self.refusal_ids], -1) - torch.logsumexp(lp[:, self.comply_ids], -1)

    @torch.no_grad()
    def extract_with_refusal(self, prompts: list[str], batch_size: int = 8,
                             steer: Optional[Steer] = None) -> tuple[np.ndarray, np.ndarray]:
        """Last-token activations ``[L+1, n, d]`` and refusal log-odds ``[n]``.

        ``steer=(layer, vec)`` adds ``vec`` to the residual stream at hidden-state index
        ``layer`` (1 = output of the first block) for every position, for the whole pass.
        """
        handle = None
        if steer is not None:
            layer, vec = steer
            v = torch.as_tensor(np.asarray(vec), dtype=next(self.model.parameters()).dtype,
                                device=self.device)

            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    return (out[0] + v,) + tuple(out[1:])
                return out + v

            handle = decoder_layers(self.model)[layer - 1].register_forward_hook(hook)
        try:
            return self._run(prompts, batch_size)
        finally:
            if handle is not None:
                handle.remove()

    def _run(self, prompts: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        texts = [self.format(p) for p in prompts]
        acts: list[np.ndarray] = []
        gaps: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                # the template already carries the special tokens when it exists
                add_special_tokens=not self.use_chat_template,
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=0)  # [L+1, B, T, D]
            # Last real token per row, whatever the padding side.
            mask = enc["attention_mask"]
            last = mask.shape[1] - 1 - mask.flip(1).argmax(1)  # index of last 1 in each row
            idx = last.view(1, -1, 1, 1).expand(hs.shape[0], -1, 1, hs.shape[-1])
            picked = hs.gather(2, idx).squeeze(2)  # [L+1, B, D]
            acts.append(picked.float().cpu().numpy())
            rows = torch.arange(len(last), device=last.device)
            gaps.append(self._refusal_logodds(out.logits[rows, last]).cpu().numpy())
        return np.concatenate(acts, axis=1), np.concatenate(gaps)
