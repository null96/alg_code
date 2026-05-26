from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal validation runner for OpenClaw local execution.")
    parser.add_argument("--data-dir", default="", help="Optional local data directory passed through by the worker.")
    parser.add_argument("--phase", default="smoke", choices=["inspect", "smoke"], help="Validation phase to run.")
    parser.add_argument("--output-json", default="", help="Optional path to write a JSON summary.")
    return parser


def _summarize_data_dir(data_dir: str) -> dict[str, object]:
    if not data_dir:
        return {"provided": False, "exists": False, "entries": []}
    path = Path(data_dir)
    entries: list[str] = []
    if path.exists() and path.is_dir():
        entries = sorted(item.name for item in path.iterdir())[:10]
    return {"provided": True, "exists": path.exists(), "entries": entries}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    payload = {
        "ok": True,
        "runner": "automl_runner",
        "phase": args.phase,
        "data_dir": _summarize_data_dir(args.data_dir),
        "message": "local execution validation succeeded",
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
