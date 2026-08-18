"""`dfine` console script.

Thin dispatcher over the existing Hydra entrypoints, plus `init` - which materializes a
`config.yaml` into the cwd so pip users get the same config-driven workflow as a clone.
Hydra overrides pass straight through: `dfine train model_name=m train.epochs=100`.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

from dfine_seg.config.resolve import CONFIG_NAME, DEFAULT_CONFIG, ENV_VAR, find_config

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

USAGE = f"""dfine <command> [hydra overrides]

  init            write a config.yaml into the current directory
  predict         run a model on an image or folder, no config needed
  demo            launch the Gradio UI (needs `pip install 'dfine-seg[demo]'`)
{_ROWS}
  version         print the installed version

Examples:
  dfine init --task segment
  dfine predict s photo.jpg -o out/
  dfine demo --port 8080
  dfine train model_name=m train.epochs=100
  dfine export export.formats=[onnx]
"""

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".npy"}


def _predict(argv: List[str]) -> int:
    """Config-free inference, so a pip user can try a model straight after installing."""
    ap = argparse.ArgumentParser(prog="dfine predict")
    ap.add_argument("model", help="size (n|s|m|l|x) or path to a .pt/.engine/.onnx/.xml/…")
    ap.add_argument("source", type=Path, help="image file or a directory of images")
    ap.add_argument("--task", choices=("detect", "segment", "sem_seg"))
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--device", help="cuda | cpu | mps (default: best available)")
    ap.add_argument("-o", "--out", type=Path, help="save annotated images / label maps here")
    args = ap.parse_args(argv)

    if args.source.is_dir():
        images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    elif args.source.is_file():
        images = [args.source]
    else:
        print(f"{args.source} not found", file=sys.stderr)
        return 1
    if not images:
        print(f"no images ({', '.join(sorted(IMAGE_EXTS))}) in {args.source}", file=sys.stderr)
        return 1

    import cv2
    import numpy as np

    from dfine_seg import load_model, read_image
    from dfine_seg.viz import Visualizer

    kwargs = {"conf_thresh": args.conf}
    if args.device:
        kwargs["device"] = args.device
    model = load_model(args.model, task=args.task, **kwargs)
    names = model.names or {}

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    # Built on the first boxed output rather than up front: `task` is a TorchModel attribute,
    # and the output itself is what actually says whether this model draws boxes or a map.
    visualizer = None

    for path in images:
        img = read_image(path)
        out = model(img, bgr=path.suffix.lower() != ".npy")[0]

        if "sem_seg" in out:
            label_map = out["sem_seg"].cpu().numpy()
            ids, counts = np.unique(label_map, return_counts=True)
            share = ", ".join(
                f"{names.get(int(i), i)} {c / label_map.size:.0%}" for i, c in zip(ids, counts)
            )
            print(f"{path.name}: {len(ids)} classes - {share}")
            if args.out:  # grayscale label map, same format as `dfine infer`
                cv2.imwrite(str(args.out / f"{path.stem}.png"), label_map)
            continue

        scores = out["scores"]
        found = ", ".join(
            f"{names.get(int(lbl), int(lbl))} {float(s):.2f}"
            for lbl, s in zip(out["labels"].cpu(), scores.cpu())
        )
        print(f"{path.name}: {len(scores)} objects{' - ' + found if len(scores) else ''}")
        if args.out:
            if visualizer is None:
                n_classes = max(names) + 1 if names else 80
                visualizer = Visualizer(n_classes=n_classes, class_names=names or None)
            drawn = visualizer.draw(img, {k: v.cpu() for k, v in out.items()})
            cv2.imwrite(str(args.out / f"{path.stem}.jpg"), drawn)

    if args.out:
        print(f"wrote {len(images)} file(s) to {args.out}")
    return 0


def _demo(argv: List[str]) -> int:
    """Launch the Gradio UI. Every model setting is changeable from the page itself."""
    ap = argparse.ArgumentParser(prog="dfine demo")
    ap.add_argument("model", nargs="?", default="s", help="model to open with (size or path)")
    ap.add_argument("--task", choices=("detect", "segment", "sem_seg"), default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="bind address; 0.0.0.0 for the LAN")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live tunnel")
    args = ap.parse_args(argv)

    try:
        from dfine_seg.demo import main as demo_main
    except ImportError as e:
        print(f"{e}. Install it with `pip install 'dfine-seg[demo]'`", file=sys.stderr)
        return 1
    demo_main(args.model, args.task, args.host, args.port, args.share)
    return 0


def _set_key(text: str, key: str, value: str) -> str:
    """Rewrite the one `key: …` line in the template, keeping its inline comment.

    A line rewrite rather than a substring replace: an exact-match replace turns into a
    silent no-op the day someone edits the template's spacing.
    """
    out, hit = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if not hit and stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            comment = stripped.partition(" #")[2]
            line = f"{indent}{key}: {value}" + (f" #{comment}" if comment else "\n")
            hit = True
        out.append(line)
    if not hit:
        raise KeyError(f"`{key}:` not found in the packaged template")
    return "".join(out)


def _init(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="dfine init")
    # A directory, not a filename: the file must be called config.yaml or nothing can
    # discover it (see config.resolve.find_config).
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
    try:
        if args.task:
            text = _set_key(text, "task", args.task)
            if args.task in ("segment", "sem_seg"):
                # Both train a MaskDecoder; the detection default leaves it at random init.
                text = _set_key(
                    text, "pretrained_model_path", "pretrained/dfine_seg_${model_name}_coco.pt"
                )
        if args.model:
            text = _set_key(text, "model_name", args.model)
    except KeyError as e:
        print(f"packaged template is malformed: {e}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    print(f"wrote {out}")
    if args.dir.resolve() != Path.cwd():
        print(f"Run commands from {args.dir}, or set {ENV_VAR}={args.dir.resolve()}")
    print("Next: edit train.root, train.label_to_name, then `dfine split && dfine train`")
    return 0


def _ddp_launch(overrides: List[str]) -> Optional[int]:
    """Mirror the Makefile: torchrun when train.ddp.enabled is set. None = not handled.

    None rather than a negative sentinel: `subprocess.call` returns -signum for a
    signal-killed child, so -1 is a real exit code here (SIGHUP).
    """
    cfg_path = find_config()
    if cfg_path is None:
        return None
    try:
        ddp = (yaml.safe_load(cfg_path.read_text()) or {}).get("train", {}).get("ddp", {}) or {}
    except Exception:
        return None
    # These same overrides go on to Hydra, so a CLI value has to win here too. Plain
    # `key=value` only - Hydra's +/~ and config groups are not reimplemented.
    enabled, n = ddp.get("enabled"), ddp.get("n_gpus", 2)
    for override in overrides:
        key, sep, val = override.partition("=")
        if not sep:
            continue
        if key == "train.ddp.enabled":
            enabled = val.strip().lower() in ("true", "1", "yes")
        elif key == "train.ddp.n_gpus":
            n = val
    if not enabled:
        return None
    try:
        n = int(n)
    except (TypeError, ValueError):
        print(f"train.ddp.n_gpus must be an integer, got {n!r}", file=sys.stderr)
        return 1
    if shutil.which("torchrun") is None:
        print("train.ddp.enabled is set but torchrun is not on PATH", file=sys.stderr)
        return 1
    print(f"Training with DDP on {n} GPUs")
    mod = COMMANDS["train"][0]
    cmd = ["torchrun", f"--nproc_per_node={n}", "--master_port=29500", "-m", mod]
    return subprocess.call(cmd + overrides)


def _run(command: str, overrides: List[str]) -> int:
    try:
        found = find_config()
    except FileNotFoundError as e:  # $DFINE_SEG_CONFIG_DIR set but empty
        print(e, file=sys.stderr)
        return 1
    if found is None:
        print(
            f"no {CONFIG_NAME}.yaml found in {Path.cwd()}.\n"
            f"Run `dfine init` to create one, or set {ENV_VAR} to a directory holding it.",
            file=sys.stderr,
        )
        return 1

    if command == "train":
        rc = _ddp_launch(overrides)
        if rc is not None:
            return rc

    from importlib import import_module

    module = import_module(COMMANDS[command][0])
    sys.argv = [f"dfine {command}", *overrides]
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
    if command == "predict":
        return _predict(rest)
    if command == "demo":
        return _demo(rest)
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
