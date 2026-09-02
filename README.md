# vllm-bounded-fs-kv-tier

A drop-in, **size-bounded** filesystem secondary tier for vLLM's KV-cache offloading —
with LRU eviction and a configurable byte cap.

## The problem

vLLM's built-in filesystem tier (`vllm/v1/kv_offload/tiering/fs/manager.py`) writes one
`.bin` file per KV block and **never removes anything**. There is no capacity limit, no
eviction, and no quota. If your `root_dir` sits on a volume you cannot afford to fill,
the cache will grow until the filesystem is full.

Audited against `vllm 0.1.dev20051+g487ecf187`:

| | finding |
|---|---|
| `FileSystemTierManager` implements | `lookup`, `submit_store`, `submit_load`, `get_finished_jobs`, `take_events`, `drain_jobs`, `on_new_request`, `on_request_finished`, `on_schedule_end`, `shutdown` |
| does **not** implement | `touch()` — so there is no recency tracking at all |
| capacity / eviction / quota | absent from the `fs` tier, the `obj` tier, and the `SecondaryTierManager` ABC |

The repo's test suite asserts these facts, so if a future vLLM grows its own capacity
handling the tests fail loudly rather than letting this quietly duplicate it.

## What this adds

* **Hard byte cap** (`max_bytes`), accepting `214748364800`, `"200GiB"` or `"200GB"`.
* **LRU eviction** down to a low-watermark (`evict_to_ratio`, default `0.9`).
* **`touch()` implemented**, so recency is real LRU rather than FIFO.
* **Restart-safe**: adopts `.bin` files left by a previous run (oldest first) and trims
  before serving, so a restart cannot stack a fresh cache on top of an old one.
* **No vLLM patching.** It loads through vLLM's supported out-of-tree extension point
  (`secondary_tiers[].module_path`), so it survives engine upgrades.

## Why eviction needs no locking

Deleting a file out from under an in-flight load is safe *by construction* in vLLM. The
built-in load task catches `OSError` (which includes `FileNotFoundError`), records how many
blocks succeeded, and `get_finished_jobs()` calls `self._lookup_manager.mark_miss(failed)`.

The affected blocks simply become a **cache miss and are recomputed** — no corruption, no
failed request. So the evictor never has to coordinate with readers.

## Accounting

`batch_store_block` writes exactly one file of `block_size` **bytes** per key, so usage is
`len(tracked_files) * block_size`. No `stat()` per file, no directory walks on the hot path.

Tracking happens at *submit* rather than completion, which deliberately over-counts a failed
store. That is self-healing: eviction tolerates `FileNotFoundError` and drops the entry.

## Install

Put `vllm_bounded_fs_tier.py` anywhere on the engine's `PYTHONPATH` (e.g. mount it at
`/opt/vllm-ext` and set `PYTHONPATH=/opt/vllm-ext`).

## Configure

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 1073741824,
    "eviction_policy": "lru",
    "secondary_tiers": [
      {
        "type": "BoundedFileSystemTierManager",
        "module_path": "vllm_bounded_fs_tier",
        "root_dir": "/kvoffload",
        "max_bytes": "200GiB",
        "evict_to_ratio": 0.9,
        "n_read_threads": 8,
        "n_write_threads": 8
      }
    ]
  }
}
```

Passed to vLLM as `--kv-transfer-config '<that json>'`.

### Options

| key | default | meaning |
|---|---|---|
| `max_bytes` | *required* | hard cap; int bytes or `"200GiB"` / `"200GB"` / `"500MB"` |
| `evict_to_ratio` | `0.9` | evict down to this fraction of the cap, so eviction is batched rather than per-block |

All other keys (`root_dir`, `n_read_threads`, `n_write_threads`, `enable_kv_events`,
`locality`) are passed straight through to vLLM's `FileSystemTierManager`.

## A gotcha worth knowing

Set **`PYTHONHASHSEED`** to a fixed value (e.g. `0`) on every engine process. vLLM seeds its
block-content hash chain with random bytes otherwise, so identical token content produces
different filenames on each run — meaning the on-disk cache is silently invalidated by every
restart, and cannot be shared between instances.

## Tests

```bash
python3 tests/test_bounded_fs_tier.py
```

Runs anywhere vLLM is importable — no GPU and no CUDA context required. Covers the parent
contract, `parse_bytes`, real-file LRU eviction, tolerance of already-deleted files, and
restart adoption.

## License

Apache-2.0, matching vLLM.
