import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _AvailableAllocator:
    def __init__(self, available_size: int):
        self._available_size = available_size

    def available_size(self) -> int:
        return self._available_size


class _FakeTokenAllocator:
    def __init__(self, logical_available_size: int, *, fail_logical_alloc: bool = False):
        self.logical_attn_allocator = _AvailableAllocator(logical_available_size)
        self.device = torch.device("cpu")
        self.page_size = 64
        self.fail_logical_alloc = fail_logical_alloc
        self.alloc_logical_only_calls = 0

    def alloc_logical_only(self, *args, extend_num_tokens: int, **kwargs):
        self.alloc_logical_only_calls += 1
        if self.fail_logical_alloc:
            return None
        return torch.arange(extend_num_tokens, dtype=torch.int64)


class _HostPool:
    def __init__(self, available_size: int, *, fail_alloc: bool = False):
        self._available_size = available_size
        self.fail_alloc = fail_alloc
        self.free_calls = 0
        self.freed_indices = []

    def available_size(self) -> int:
        return self._available_size

    @staticmethod
    def _round_up_to_page_size(size: int, page_size: int = 64) -> int:
        return (size + page_size - 1) // page_size * page_size

    def alloc_paged_token_slots(
        self,
        req_to_host_pool,
        req_to_host_pool_allocated_len,
        req_pool_idx,
        start_pos,
        num_tokens,
    ):
        if self.fail_alloc:
            raise RuntimeError("HiSparse host mem pool alloc failed for 1 pages")

        end_pos = start_pos + num_tokens
        rounded_len = self._round_up_to_page_size(end_pos)
        if rounded_len > self._available_size:
            raise RuntimeError("HiSparse host mem pool alloc failed for 1 pages")

        host_indices = torch.arange(rounded_len, dtype=torch.int64)
        req_to_host_pool[req_pool_idx, :rounded_len] = host_indices
        req_to_host_pool_allocated_len[req_pool_idx] = rounded_len
        self._available_size -= rounded_len
        return req_to_host_pool[req_pool_idx, start_pos:end_pos]

    def allocated_host_indices(
        self, req_to_host_pool, req_pool_idx: int, allocated_len: int
    ):
        host_indices = req_to_host_pool[req_pool_idx, :allocated_len]
        return host_indices[host_indices >= 0]

    def free(self, indices: torch.Tensor) -> int:
        self.free_calls += 1
        self.freed_indices.append(indices.clone())
        self._available_size += indices.numel()
        return indices.numel()


def _make_req(fill_len: int):
    req = SimpleNamespace(
        rid="hisparse-prealloc-test",
        origin_input_ids=list(range(fill_len)),
        output_ids=[],
        fill_ids=list(range(fill_len)),
        req_pool_idx=None,
        kv_allocated_len=0,
        kv_committed_len=0,
        inflight_middle_chunks=0,
    )
    req.set_extend_input_len = lambda extend_input_len: setattr(
        req, "extend_input_len", extend_input_len
    )
    return req


def _make_queue(
    *,
    host_pool: _HostPool,
    logical_available_size: int = 1024,
    fail_logical_alloc: bool = False,
):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.req_to_token_pool = ReqToTokenPool(
        size=1,
        max_context_len=256,
        device="cpu",
        enable_memory_saver=False,
    )
    token_allocator = _FakeTokenAllocator(
        logical_available_size, fail_logical_alloc=fail_logical_alloc
    )
    queue.token_to_kv_pool_allocator = token_allocator
    queue.tree_cache = SimpleNamespace(evictable_size=lambda: 0, protected_size=lambda: 0)
    queue.transfer_queue = SimpleNamespace(queue=[])
    queue.retracted_queue = []
    queue.num_reserved_decode_tokens = 0

    coordinator = SimpleNamespace(
        mem_pool_host=host_pool,
        req_to_host_pool=torch.full((2, 256), -1, dtype=torch.int64),
        req_to_host_pool_allocated_len=torch.zeros((2,), dtype=torch.int64),
    )
    queue.scheduler = SimpleNamespace(
        enable_hisparse=True,
        hisparse_coordinator=coordinator,
        server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        last_batch=None,
    )
    return queue, token_allocator, coordinator


class TestHiSparseDecodePrealloc(CustomTestCase):
    def test_allocatable_budget_is_capped_by_host_pool(self):
        queue, _, _ = _make_queue(
            host_pool=_HostPool(available_size=128),
            logical_available_size=1024,
        )

        self.assertEqual(queue._allocatable_token_budgets(), 128)

    def test_host_alloc_failure_rolls_back_req_slot(self):
        queue, token_allocator, _ = _make_queue(
            host_pool=_HostPool(available_size=0, fail_alloc=True)
        )
        req = _make_req(fill_len=64)
        initial_req_slots = queue.req_to_token_pool.available_size()

        with self.assertRaisesRegex(RuntimeError, "host mem pool alloc failed"):
            queue._pre_alloc(req)

        self.assertEqual(queue.req_to_token_pool.available_size(), initial_req_slots)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(req.kv_allocated_len, 0)
        self.assertEqual(req.kv_committed_len, 0)
        self.assertEqual(token_allocator.alloc_logical_only_calls, 0)

    def test_logical_alloc_failure_rolls_back_host_and_req_slot(self):
        host_pool = _HostPool(available_size=128)
        queue, _, coordinator = _make_queue(
            host_pool=host_pool,
            fail_logical_alloc=True,
        )
        req = _make_req(fill_len=64)
        initial_req_slots = queue.req_to_token_pool.available_size()
        initial_host_available = host_pool.available_size()

        with self.assertRaisesRegex(RuntimeError, "logical KV allocation failed"):
            queue._pre_alloc(req)

        self.assertEqual(queue.req_to_token_pool.available_size(), initial_req_slots)
        self.assertEqual(host_pool.available_size(), initial_host_available)
        self.assertEqual(host_pool.free_calls, 1)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(req.kv_allocated_len, 0)
        self.assertEqual(req.kv_committed_len, 0)
        self.assertTrue(torch.all(coordinator.req_to_host_pool == -1))
        self.assertTrue(torch.all(coordinator.req_to_host_pool_allocated_len == 0))


if __name__ == "__main__":
    unittest.main()
