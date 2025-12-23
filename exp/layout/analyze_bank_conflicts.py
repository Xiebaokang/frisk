#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Sequence


def default_tool_path() -> str:
  return os.path.join("build", "exp", "layout", "frisk_layout_tool")


def parse_shape(value: str) -> Sequence[int]:
  parts = value.split(",")
  if len(parts) != 3:
    raise argparse.ArgumentTypeError("shape must be M,N,K")
  try:
    return [int(part) for part in parts]
  except ValueError as err:
    raise argparse.ArgumentTypeError(str(err)) from err


def run_tool(args: argparse.Namespace) -> Dict[str, Any]:
  tool = shutil.which(args.tool) or args.tool
  if not os.path.exists(tool):
    raise FileNotFoundError(f"layout tool not found: {tool}")
  cmd = [
      tool,
      f"--target={args.target}",
      f"--block-m={args.shape[0]}",
      f"--block-n={args.shape[1]}",
      f"--block-k={args.shape[2]}",
      f"--dtype={args.dtype}",
      f"--threads={args.threads}",
      f"--a-space={args.a_space}",
      f"--b-space={args.b_space}",
      f"--c-space={args.c_space}",
      f"--lane-vector={max(1, args.lane_vector)}",
  ]
  if args.grid:
    cmd.append(f"--grid={','.join(str(dim) for dim in args.grid)}")
  if args.trans_a:
    cmd.append("--trans-a")
  if args.trans_b:
    cmd.append("--trans-b")
  if args.ldmatrix:
    cmd.append("--ldmatrix")

  completed = subprocess.run(
      cmd, check=True, stdout=subprocess.PIPE, stderr=sys.stderr.buffer
  )
  return json.loads(completed.stdout.decode("utf-8"))


def print_config(config: Dict[str, Any]) -> None:
  print("== Config ==")
  print(f"target      : {config.get('target')}")
  block = config.get("block_shape", {})
  print(
      f"shape       : M={block.get('M')} "
      f"N={block.get('N')} K={block.get('K')}"
  )
  print(f"dtype       : {config.get('dtype')}")
  print(f"threads     : {config.get('threads')}")
  if "lane_vector" in config:
    suffix = " (ldmatrix)" if config.get("ldmatrix_like") else ""
    print(f"lane vector : {config.get('lane_vector')}{suffix}")
  if config.get("lane_vector"):
    print(f"lane vector : {config.get('lane_vector')} (ldmatrix={config.get('ldmatrix_like')})")
  grid = config.get("grid", [])
  if grid:
    print(f"grid        : {list(grid)}")
  spaces = config.get("memory_spaces", {})
  if spaces:
    print("spaces      :", ", ".join(f"{k}={v}" for k, v in spaces.items()))
  print()


def format_index(entry: Dict[str, Any]) -> str:
  index = entry.get("forward_index")
  if not index:
    return "<none>"
  return index.replace("\n", "")


def describe_operand(entry: Dict[str, Any]) -> None:
  operand = entry.get("operand", "?")
  print(f"Operand {operand} ({entry.get('memory_space','?')}):")
  layout_str = format_index(entry)
  print(f"  layout index     : {layout_str}")
  bank_stats = entry.get("bank_stats")
  baseline = entry.get("baseline_row_major")
  if bank_stats:
    print(
        "  bank stats       : "
        f"avg_max={bank_stats.get('avg_max_conflict', 'n/a'):.2f}  "
        f"max={bank_stats.get('max_conflict', 'n/a')}"
    )
  if baseline:
    print(
        "  baseline (torch) : "
        f"avg_max={baseline.get('avg_max_conflict', 'n/a'):.2f}  "
        f"max={baseline.get('max_conflict', 'n/a')}"
    )
  print()


def try_torch_benchmark(args: argparse.Namespace) -> None:
  try:
    import torch  # type: ignore
  except Exception as exc:  # pragma: no cover - optional dependency
    print(f"PyTorch unavailable ({exc}); skipping runtime baseline.")
    return

  dtype_map = {
      "fp16": torch.float16,
      "bf16": torch.bfloat16,
      "fp32": torch.float32,
      "fp64": torch.float64,
  }
  torch_dtype = dtype_map.get(args.dtype.lower())
  if torch_dtype is None:
    print(f"PyTorch benchmark skipped: dtype {args.dtype} unsupported.")
    return

  device = "cuda" if torch.cuda.is_available() else "cpu"
  try:
    a = torch.randn(
        (args.shape[0], args.shape[2]), device=device, dtype=torch_dtype
    )
    b = torch.randn(
        (args.shape[2], args.shape[1]), device=device, dtype=torch_dtype
    )
  except RuntimeError as exc:
    print(f"PyTorch benchmark skipped: {exc}")
    return

  def run_once() -> None:
    torch.matmul(a, b)

  warmup = 5
  iters = 20
  if device == "cuda":
    torch.cuda.synchronize()
    for _ in range(warmup):
      run_once()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times: List[float] = []
    for _ in range(iters):
      start_event.record()
      run_once()
      end_event.record()
      torch.cuda.synchronize()
      times.append(start_event.elapsed_time(end_event) / 1e3)
  else:
    for _ in range(warmup):
      run_once()
    times = []
    for _ in range(iters):
      torch.cuda.synchronize() if torch.cuda.is_available() else None
      start = time.perf_counter()
      run_once()
      end = time.perf_counter()
      times.append(end - start)

  avg_sec = sum(times) / len(times)
  flops = 2.0 * args.shape[0] * args.shape[1] * args.shape[2]
  tflops = flops / avg_sec / 1e12
  device_name = torch.cuda.get_device_name(0) if device == "cuda" else device
  print(
      f"PyTorch baseline throughput: {tflops:.2f} TFLOPS (device: {device_name})"
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Analyze frisk.gemm shared-memory layouts "
      "and compare against PyTorch row-major baseline."
  )
  parser.add_argument(
      "--tool",
      default=default_tool_path(),
      help="Path to frisk_layout_tool (default: %(default)s)",
  )
  parser.add_argument("--target", default="sm_80", help="Target arch (e.g. sm_80)")
  parser.add_argument(
      "--shape",
      type=parse_shape,
      default=[128, 128, 64],
      help="Tile shape as M,N,K (default: %(default)s)",
  )
  parser.add_argument(
      "--dtype", default="fp16", help="Element dtype (fp16, bf16, fp32, ...)"
  )
  parser.add_argument(
      "--threads", type=int, default=128, help="Threads per block (default: %(default)s)"
  )
  parser.add_argument(
      "--lane-vector",
      type=int,
      default=1,
      help="Elements loaded per lane (default: %(default)s)",
  )
  parser.add_argument(
      "--ldmatrix",
      action="store_true",
      help="Approximate CUDA ldmatrix.x4 pattern (forces lane-vector>=4)",
  )
  parser.add_argument(
      "--grid",
      type=lambda s: [int(dim) for dim in s.split(",")],
      default=None,
      help="Grid dimensions as comma-separated ints (optional)",
  )
  parser.add_argument("--a-space", default="shared", help="Memory space for operand A")
  parser.add_argument("--b-space", default="shared", help="Memory space for operand B")
  parser.add_argument("--c-space", default="local", help="Memory space for operand C")
  parser.add_argument("--trans-a", action="store_true", help="Transpose A")
  parser.add_argument("--trans-b", action="store_true", help="Transpose B")
  args = parser.parse_args()

  result = run_tool(args)
  config = result.get("config", {})
  print_config(config)
  for entry in result.get("layouts", []):
    describe_operand(entry)
  try_torch_benchmark(args)


if __name__ == "__main__":
  main()
