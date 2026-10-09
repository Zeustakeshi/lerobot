"""Synchronous TurboVLA client for a local or remote ``vla.cpp`` server."""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

import torch
from torch import Tensor

from lerobot.utils.constants import OBS_LANGUAGE, OBS_STATE
from lerobot.utils.import_utils import require_package

from . import vla_pb2
from .configuration_turbovla import TurboVLAConfig


class VlaCppTurboVLAClient:
    """Send raw observations to vla-server and return world-unit action chunks.

    The server owns image/state normalization and action de-normalization.  This
    deliberately makes the checkpoint, rather than a robot-specific Python
    adapter, the source of truth for those transforms.
    """

    def __init__(self, config: TurboVLAConfig) -> None:
        require_package("transformers", extra="turbovla")
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("TurboVLA vla.cpp backend requires pyzmq; install lerobot[pyzmq-dep]") from exc
        from transformers import AutoTokenizer

        self.config = config
        self._zmq = zmq
        self._context = zmq.Context.instance()
        # A ZeroMQ socket has thread affinity.  The async rollout engine creates
        # this policy on the control thread but performs every request on its
        # worker thread, so defer socket creation until the first request.
        self._socket = None
        self._socket_lock = Lock()
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.language_encoder_id,
            revision=config.language_encoder_revision,
            local_files_only=config.local_files_only,
            use_fast=True,
        )
        self._request_id = 0

    def close(self) -> None:
        with self._socket_lock:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None

    def _get_socket(self):
        """Create the REQ socket on the thread that will use it."""
        with self._socket_lock:
            if self._socket is None:
                socket = self._context.socket(self._zmq.REQ)
                socket.setsockopt(self._zmq.LINGER, 0)
                timeout = int(self.config.vla_cpp_timeout_ms)
                socket.setsockopt(self._zmq.RCVTIMEO, timeout)
                socket.setsockopt(self._zmq.SNDTIMEO, timeout)
                socket.connect(self.config.vla_cpp_addr)
                self._socket = socket
            return self._socket

    def _instructions(self, language: object, batch_size: int) -> list[str]:
        if isinstance(language, str):
            values = [language]
        elif isinstance(language, Sequence) and all(isinstance(item, str) for item in language):
            values = list(language)
        else:
            raise TypeError(f"{OBS_LANGUAGE!r} must be a string or sequence of strings")
        if len(values) != batch_size or any(not value.strip() for value in values):
            raise ValueError("TurboVLA vla.cpp backend requires one non-empty instruction per observation")
        return values

    @staticmethod
    def _rgb_u8(image: Tensor, key: str) -> tuple[int, int, bytes]:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{key!r} must have shape [3,H,W], got {tuple(image.shape)}")
        image = image.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(image).all() or image.numel() == 0:
            raise ValueError(f"{key!r} has no finite RGB pixels")
        if image.amin() < 0 or image.amax() > 1:
            raise ValueError(f"{key!r} must be raw RGB in [0,1]; preprocessing was not configured for vla.cpp")
        hwc = image.mul(255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
        height, width = hwc.shape[:2]
        return int(height), int(width), hwc.numpy().tobytes()

    @torch.no_grad()
    def predict(self, batch: dict[str, object]) -> Tensor:
        required = (*self.config.image_keys, OBS_STATE, OBS_LANGUAGE)
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"TurboVLA vla.cpp input is missing {missing}")
        first = batch[self.config.image_keys[0]]
        if not isinstance(first, Tensor) or first.ndim != 4:
            raise ValueError("TurboVLA vla.cpp images must be batched [B,3,H,W]")
        batch_size = int(first.shape[0])
        if batch_size != 1:
            raise ValueError("vla-server currently serves one rollout observation per request (B=1)")
        instructions = self._instructions(batch[OBS_LANGUAGE], batch_size)
        state = batch[OBS_STATE]
        if not isinstance(state, Tensor) or tuple(state.shape) != (1, self.config.state_dim):
            raise ValueError(f"{OBS_STATE!r} must have shape [1,{self.config.state_dim}]")
        state_values = state.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if not torch.isfinite(state_values).all():
            raise ValueError(f"{OBS_STATE!r} contains NaN or Inf")

        tokenized = self._tokenizer(
            instructions[0],
            truncation=True,
            max_length=self.config.max_text_length,
            padding="max_length" if self.config.text_padding_length is not None else False,
            return_attention_mask=True,
        )
        token_ids = tokenized["input_ids"]
        attention = tokenized["attention_mask"]
        if self.config.text_padding_length is not None:
            target = self.config.text_padding_length
            token_ids = token_ids[:target] + [self._tokenizer.pad_token_id] * max(0, target - len(token_ids))
            attention = attention[:target] + [0] * max(0, target - len(attention))

        request = vla_pb2.PredictRequest(request_id=self._request_id)
        self._request_id += 1
        for key in self.config.image_keys:
            image = batch[key]
            if not isinstance(image, Tensor) or image.ndim != 4 or image.shape[0] != 1:
                raise ValueError(f"{key!r} must have shape [1,3,H,W]")
            height, width, data = self._rgb_u8(image[0], key)
            request.images.add(encoding=vla_pb2.Image.RGB_U8, height=height, width=width, data=data)
        request.lang_tokens.extend(int(token) for token in token_ids)
        request.attention_mask.extend(int(mask) for mask in attention)
        request.state.extend(float(value) for value in state_values.tolist())

        try:
            socket = self._get_socket()
            socket.send(request.SerializeToString())
            payload = socket.recv()
        except self._zmq.Again as exc:
            # A REQ socket that timed out receiving cannot send another
            # request.  Recreate it for the async worker's retry.
            self.close()
            raise TimeoutError(
                f"No response from vla-server at {self.config.vla_cpp_addr} within "
                f"{self.config.vla_cpp_timeout_ms} ms"
            ) from exc
        response = vla_pb2.PredictResponse()
        response.ParseFromString(payload)
        if response.error:
            raise RuntimeError(f"vla-server rejected request: {response.error}")
        if response.chunk_size != self.config.chunk_size or response.action_dim != self.config.action_dim:
            raise RuntimeError(
                "vla-server/checkpoint shape mismatch: "
                f"got [{response.chunk_size},{response.action_dim}], expected "
                f"[{self.config.chunk_size},{self.config.action_dim}]"
            )
        actions = torch.tensor(response.action_chunk, dtype=torch.float32).reshape(
            1, response.chunk_size, response.action_dim
        )
        return actions
