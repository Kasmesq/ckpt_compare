import torch
from datastates.ckpt.src import handle as datastates_handle
import time
import sys
import os
from datastates.utils import get_logger


# Acts as a Python Interface to manage the CPP checkpoint engine
class CkptEngine:
    def __init__(self, host_cache_size, gpu_device_id, rank) -> None:
        try:
            self.ckpt_engine = datastates_handle(host_cache_size, gpu_device_id, rank)
            self.logger = get_logger(__name__)
            self.last_ckpt_version = -1
        except Exception as exc:
            print(f"[DataStates.ckpt][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    # This function accepts a list of tuples containing tensors to checkpoint.
    # Each tuple contains: version, torch.Tensor, file offset, and path
    def async_save(self, tensors: list[tuple[int, torch.Tensor, int, str]]):
        try:
            for t in tensors:
                version, tensor, file_offset, path = t
                tensor_bytes = tensor.numel() * tensor.element_size()
                assert tensor_bytes > 0, "Tensor size should be > 0"
                self.ckpt_engine.ckpt_tensor(
                    version,
                    tensor,
                    tensor_bytes,
                    file_offset,
                    path,
                )
        except Exception as exc:
            self.logger.error(f"[DataStates.ckpt][ERROR][async_save] {exc}")
            sys.exit(-1)

    def load(self, tensors: list[tuple[int, torch.Tensor, int, str]]):
        try:
            for t in tensors:
                version, tensor, file_offset, path = t
                file_size = os.path.getsize(path)
                tensor_bytes = tensor.numel() * tensor.element_size()
                assert tensor_bytes > 0, "Tensor size should be > 0"
                assert (
                    file_offset + tensor_bytes <= file_size
                ), f"Tensor at offset {file_offset} overflows file size {file_size}"

                self.ckpt_engine.restore_tensor(
                    version,
                    tensor,
                    tensor_bytes,
                    file_offset,
                    path,
                )
                self.logger.info(
                    f"[DataStates.ckpt] Restored tensor {tensor_bytes} from {file_offset}"
                )
        except Exception as exc:
            self.logger.error(f"[DataStates.ckpt][ERROR][load] {exc}")
            sys.exit(-1)

    def commit(self, tag):
        self.wait()
        self.logger.info(f"[DataStates.ckpt] Checkpoint {tag} is ready now!")
        self.last_ckpt_version += 1
        return True

    def wait(self):
        try:
            if getattr(self, "ckpt_engine", None) is None:
                return 0.0
            t0 = time.time()
            result = self.ckpt_engine.wait()
            elapsed = time.time() - t0

            self.logger.info(
                f"[DataStates.ckpt] wait() done | elapsed={elapsed:.4f}s"
            )

            # C++가 void면 elapsed를 반환하고,
            # 나중에 C++에서 값을 반환하면 그 값을 우선 사용
            return result if result is not None else elapsed
        except Exception as exc:
            self.logger.error(f"[DataStates.ckpt][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)

    def has_wait_durable(self):
        try:
            fn = getattr(self.ckpt_engine, "wait_durable", None)
            available = callable(fn)
            self.logger.info(
                f"[DataStates.ckpt] has_wait_durable={int(available)}"
            )
            return available
        except Exception as exc:
            self.logger.error(
                f"[DataStates.ckpt][ERROR] From has_wait_durable, generated exception: {exc}"
            )
            return False

    def wait_durable(self):
        try:
            if getattr(self, "ckpt_engine", None) is None:
                return 0.0
            fn = getattr(self.ckpt_engine, "wait_durable", None)
            if not callable(fn):
                raise RuntimeError(
                    "[DataStates.ckpt] wait_durable() is not available on the CPP engine"
                )

            t0 = time.time()
            result = fn()
            elapsed = time.time() - t0

            self.logger.info(
                f"[DataStates.ckpt] wait_durable() done | elapsed={elapsed:.4f}s"
            )

            # C++가 void면 elapsed 반환
            return result if result is not None else elapsed
        except Exception as exc:
            self.logger.error(
                f"[DataStates.ckpt][ERROR] From wait_durable, generated exception: {exc}"
            )
            sys.exit(-1)

    def close(self):
        try:
            if getattr(self, "ckpt_engine", None) is not None:
                self.ckpt_engine.shutdown()
                self.ckpt_engine = None
                self.logger.info("[DataStates.ckpt] shutdown() done")
        except Exception as exc:
            self.logger.error(
                f"[DataStates.ckpt][ERROR] From close, generated exception: {exc}"
            )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass