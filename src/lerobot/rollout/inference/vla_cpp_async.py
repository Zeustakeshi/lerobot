"""Asynchronous action-chunk execution for a ``vla.cpp``-backed policy.

Unlike Real-Time Chunking (RTC), this engine never conditions a model on an
old action prefix and never blends chunks.  It simply overlaps request N+1
with execution of N, as recommended by vla.cpp's async client example.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections import deque
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S = 0.005
_ERROR_RETRY_DELAY_S = 0.5
_MAX_CONSECUTIVE_ERRORS = 5
_JOIN_TIMEOUT_S = 3.0


class VlaCppAsyncInferenceEngine(InferenceEngine):
    """Execute fresh vla.cpp action chunks on one worker without RTC semantics.

    The worker is the sole caller of ``predict_action_chunk``.  This is
    essential for pyzmq because the vla.cpp REQ socket is not thread-safe.
    TurboVLA emits absolute targets, not deltas.  Predicting a replacement
    chunk from an observation captured while old targets remain queued makes
    that response stale before it can be dispatched and can pull a robot back
    toward an earlier pose.  Therefore a request starts only after the queue
    drains.  Inference still runs outside the control thread.
    """

    def __init__(
        self,
        *,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        hw_features: dict,
        task: str,
        device: str | None,
        queue_threshold: int,
        shutdown_event: Event | None = None,
    ) -> None:
        super().__init__(task=task)
        if queue_threshold < 0:
            raise ValueError("vla_cpp_async.queue_threshold must be non-negative")
        chunk_size = int(getattr(policy.config, "chunk_size", 0))
        if chunk_size < 1:
            raise ValueError("vla_cpp_async requires a policy with a positive chunk_size")
        if queue_threshold != 0:
            logger.warning(
                "vla_cpp_async queue_threshold=%d is unsafe for absolute action chunks; using 0",
                queue_threshold,
            )
            queue_threshold = 0

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._hw_features = hw_features
        self._device = torch.device(device or "cpu")
        self._queue_threshold = queue_threshold
        self._actions: deque[tuple[torch.Tensor, str]] = deque()
        self._actions_lock = Lock()
        self._obs_holder: dict[str, Any] = {"obs": None}
        self._obs_lock = Lock()
        self._reset_epoch = 0
        self._active = Event()
        self._shutdown = Event()
        self._failed = Event()
        self._failure_traceback: str | None = None
        self._global_shutdown_event = shutdown_event
        self._thread: Thread | None = None

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def failure_traceback(self) -> str | None:
        return self._failure_traceback

    def start(self) -> None:
        self._shutdown.clear()
        self._thread = Thread(target=self._loop, daemon=True, name="VlaCppAsyncInference")
        self._thread.start()
        logger.info("vla.cpp async inference thread started (queue_threshold=%d)", self._queue_threshold)

    def stop(self) -> None:
        self._shutdown.set()
        self._active.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("vla.cpp async thread did not join within %.1fs", _JOIN_TIMEOUT_S)
        self._thread = None

    def pause(self) -> None:
        self._active.clear()

    def resume(self) -> None:
        self._active.set()

    def reset(self) -> None:
        # A reset is normally called while rollout is paused.  The epoch still
        # guarantees an in-flight response cannot re-enter the fresh queue.
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        with self._obs_lock:
            self._obs_holder["obs"] = None
            self._reset_epoch += 1
        with self._actions_lock:
            self._actions.clear()
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        del obs_frame
        with self._actions_lock:
            if not self._actions:
                return None
            action, task = self._actions.popleft()
        self._set_dispatched_task(task)
        return action

    def notify_observation(self, obs: dict) -> None:
        # The rollout creates a fresh processed observation every control tick.
        # Retaining the latest one is therefore safe and avoids copying camera
        # frames on the real-time control thread.
        with self._obs_lock:
            self._obs_holder["obs"] = obs

    @property
    def control_thread_owns_policy(self) -> bool:
        # vla-server currently exposes action prediction only, not TurboVLA's
        # text-generation head.
        return False

    def _queue_size(self) -> int:
        with self._actions_lock:
            return len(self._actions)

    def _clear_actions(self) -> None:
        with self._actions_lock:
            self._actions.clear()

    def _append_chunk(self, actions: torch.Tensor, task: str) -> None:
        if actions.ndim != 2:
            raise RuntimeError(f"vla.cpp action chunk must be [T,A], got {tuple(actions.shape)}")
        with self._actions_lock:
            self._actions.extend((action.detach().cpu().clone(), task) for action in actions)

    def _loop(self) -> None:
        try:
            consecutive_errors = 0
            while not self._shutdown.is_set():
                if not self._active.is_set():
                    time.sleep(_IDLE_SLEEP_S)
                    continue

                task, changed = self._take_task()
                if changed:
                    # Do not execute stale instructions.  This differs
                    # deliberately from RTC: no prefix can be merged safely.
                    self._clear_actions()
                    logger.info("Task changed to '%s' — dropped stale vla.cpp actions", task)

                queue_size_at_request = self._queue_size()
                if queue_size_at_request > self._queue_threshold:
                    time.sleep(_IDLE_SLEEP_S)
                    continue

                with self._obs_lock:
                    obs = self._obs_holder["obs"]
                    epoch = self._reset_epoch
                if obs is None:
                    time.sleep(_IDLE_SLEEP_S)
                    continue

                try:
                    started = time.perf_counter()
                    obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                    obs_batch = prepare_observation_for_inference(
                        obs_batch, self._device, task, self._robot.robot_type
                    )
                    preprocessed = self._preprocessor(obs_batch)
                    raw_actions = self._policy.predict_action_chunk(preprocessed)
                    actions = self._postprocessor(raw_actions).squeeze(0)
                    if not isinstance(actions, torch.Tensor):
                        raise RuntimeError("postprocessor did not return a tensor action chunk")
                    with self._obs_lock:
                        epoch_unchanged = epoch == self._reset_epoch
                    # A task switch can arrive while vla-server is computing.
                    # Drop that old response before the control loop can pop it.
                    task_unchanged = self.task == task
                    # A chunk predicted from an earlier observation must not
                    # start after the preceding queue has already drained: it
                    # would command the arm back toward the old pose.  This is
                    # the non-RTC equivalent of an elapsed-prefix guard.  It
                    # intentionally trades a short hold for a safe, fresh
                    # request when vla-server inference is slower than the
                    # remaining action horizon.  The first chunk (queue was
                    # empty at submit time) is, of course, still accepted.
                    queue_drained_during_request = queue_size_at_request > 0 and self._queue_size() == 0
                    if epoch_unchanged and task_unchanged and not queue_drained_during_request:
                        self._append_chunk(actions, task)
                    elif queue_drained_during_request:
                        logger.warning(
                            "Discarding stale vla.cpp prefetch: %d queued actions drained during %.3fs inference",
                            queue_size_at_request,
                            time.perf_counter() - started,
                        )
                    else:
                        logger.info("Discarding stale vla.cpp action chunk (reset or task changed during inference)")
                    consecutive_errors = 0
                    logger.debug(
                        "vla.cpp async inference latency=%.3fs, queue=%d",
                        time.perf_counter() - started,
                        self._queue_size(),
                    )
                except Exception as exc:
                    consecutive_errors += 1
                    logger.error(
                        "vla.cpp async inference error (%d/%d): %s",
                        consecutive_errors,
                        _MAX_CONSECUTIVE_ERRORS,
                        exc,
                    )
                    logger.debug(traceback.format_exc())
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise
                    time.sleep(_ERROR_RETRY_DELAY_S)
        except Exception:
            self._failure_traceback = traceback.format_exc()
            logger.error("Fatal error in vla.cpp async thread:\n%s", self._failure_traceback)
            self._failed.set()
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
        finally:
            # Close the ZMQ socket on its owner thread.  pyzmq sockets must
            # not be created on the control thread then closed there.
            client = getattr(self._policy, "_vla_cpp_client", None)
            if client is not None:
                client.close()
