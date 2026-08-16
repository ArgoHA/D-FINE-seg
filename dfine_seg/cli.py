"""`dfine-seg` console script.

Thin dispatcher over the existing Hydra entrypoints, plus `init` — which materializes a
`config.yaml` into the cwd so pip users get the same config-driven workflow as a clone.
Hydra overrides pass straight through: `dfine-seg train model_name=m train.epochs=100`.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

from dfine_seg._config import CONFIG_NAME, DEFAULT_CONFIG, ENV_VAR, find_config

# subcommand -> (module providing a Hydra-decorated main(), help line)
COMMANDS = {
    "split": ("dfine_seg.etl.split", "split the dataset into train/val(/test)"),
    "train": ("dfine_seg.dl.train", "train a model"),
    "export": ("dfine_seg.dl.export", "export to onnx / tensorrt / openvino / coreml"),
    "bench": ("dfine_seg.dl.bench", "benchmark exported backends against ground truth"),
    "infer": ("dfine_seg.dl.infer", "run inference over a folder of images or videos"),
    "check-errors": ("dfine_seg.dl.check_errors", "dump FP/FN mismatches against ground truth"),
    "test-batching": ("dfine_seg.dl.test_batching", "sweep batch sizes"),
    "ov-int8": ("dfine_seg.dl.ov_int8", "OpenVINO INT8 quantization"),
    "trt-int8": ("dfine_seg.dl.trt_int8", "TensorRT INT8 calibration"),
}

_ROWS = "\n".join(f"  {c:<15} {h}" for c, (_, h) in COMMANDS.items())

USAGE = f"""dfine-seg <command> [hydra overrides]

  init            write a config.yaml into the current directory
{_ROWS}
  version         print the installed version

Examples:
  dfine-seg init --task segment
  dfine-seg train model_name=m train.epochs=100
  dfine-seg export export.formats=[onnx]
"""


def _init(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="dfine-seg init")
    # A directory, not a filename: the file must be called config.yaml or nothing can
    # discover it (see _config.find_config).
    ap.add_argument("-d", "--dir", type=Path, default=Path("."), help="where to write it")
    ap.add_argument("--task", choices=("detect", "segment", "sem_seg"))
    ap.add_argument("--model", choices=("n", "s", "m", "l", "x"))
    ap.add_argument("-f", "--force", action="store_true", help="overwrite an existing config")
    args = ap.parse_args(argv)

    out = args.dir / f"{CONFIG_NAME}.yaml"
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    if not DEFAULT_CONFIG.is_file():
        print(f"packaged template missing: {DEFAULT_CONFIG}", file=sys.stderr)
        return 1

    text = DEFAULT_CONFIG.read_text()
    if args.task:
        text = text.replace("\ntask: detect ", f"\ntask: {args.task} ", 1)
    if args.model:
        text = text.replace("\nmodel_name: s ", f"\nmodel_name: {args.model} ", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    print(f"wrote {out}")
    if args.dir.resolve() != Path.cwd():
        print(f"Run commands from {args.dir}, or set {ENV_VAR}={args.dir.resolve()}")
    print("Next: edit train.root, train.label_to_name, then `dfine-seg split && dfine-seg train`")
    return 0


def _ddp_launch(overrides: List[str]) -> int:
    """Mirror the Makefile: torchrun when train.ddp.enabled is set."""
    cfg_path = find_config()
    if cfg_path is None:
        return -1
    try:
        ddp = (yaml.safe_load(cfg_path.read_text()) or {}).get("train", {}).get("ddp", {})
    except Exception:
        return -1
    if not ddp.get("enabled"):
        return -1
    n = int(ddp.get("n_gpus", 2))
    if shutil.which("torchrun") is None:
        print("train.ddp.enabled is set but torchrun is not on PATH", file=sys.stderr)
        return 1
    print(f"Training with DDP on {n} GPUs")
    mod = COMMANDS["train"][0]
    cmd = ["torchrun", f"--nproc_per_node={n}", "--master_port=29500", "-m", mod]
    return subprocess.call(cmd + overrides)


def _run(command: str, overrides: List[str]) -> int:
    if find_config() is None:
        print(
            f"no {CONFIG_NAME}.yaml found in {Path.cwd()}.\n"
            f"Run `dfine-seg init` to create one, or set {ENV_VAR} to a directory holding it.",
            file=sys.stderr,
        )
        return 1

    if command == "train":
        rc = _ddp_launch(overrides)
        if rc >= 0:
            return rc

    from importlib import import_module

    module = import_module(COMMANDS[command][0])
    sys.argv = [f"dfine-seg {command}", *overrides]
    module.main()  # @hydra.main parses sys.argv[1:]
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    command, rest = argv[0], argv[1:]

    if command == "init":
        return _init(rest)
    if command in ("version", "--version", "-V"):
        from dfine_seg import __version__

        print(__version__)
        return 0
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    return _run(command, rest)


if __name__ == "__main__":
    sys.exit(main())
