from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ._brain_common import ENERGY_MODES
from ._cli_chat import cmd_chat, cmd_start
from ._cli_commands import (
    cmd_ask,
    cmd_ask_batch,
    cmd_ask_stdin,
    cmd_explain,
    cmd_forget,
    cmd_inspect,
    cmd_recall,
    cmd_status,
    cmd_train,
)
from ._cli_common import DEFAULT_BRAIN_DIR, brain_dir_from_load_path, normalize_default_argv

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puhl-luck",
        description="Train, load, chat, and rank with a local unified brain memory.",
        epilog=(
            "short use: puhl-luck | puhl-luck brain_data | "
            "puhl-luck brain_data\\brain_memory.pkl | puhl-luck \"question\" \"A,B\""
        ),
    )
    parser.add_argument("--brain-dir", default=str(DEFAULT_BRAIN_DIR), help="learned brain folder")
    parser.add_argument("--window-size", type=int, default=12, help="event co-activation window")
    sub = parser.add_subparsers(dest="command")

    start = sub.add_parser("start", aliases=["s"], help="auto-train data folder and open chat")
    start.add_argument("data_path", nargs="?", help="file or folder to auto-train before chat")
    start.add_argument("--data", help="file or folder to auto-train before chat; default is data")
    start.add_argument("--epochs", type=int, default=1, help="number of replay passes")
    start.add_argument("--micro-hash-bits", type=int, choices=[16, 32], default=16, help="micro rank feature hash width")
    start_learn_group = start.add_mutually_exclusive_group()
    start_learn_group.add_argument("--learn", dest="learn", action="store_true", default=True, help="learn from chat turns")
    start_learn_group.add_argument("--no-learn", dest="learn", action="store_false", help="do not learn from chat turns")
    start.add_argument("--max-new-tokens", type=int, default=24)
    start.set_defaults(func=cmd_start)

    train = sub.add_parser("train", aliases=["t", "learn"], help="learn text, image, audio, and byte events")
    train.add_argument("data_path", nargs="?", help="file or folder with supported training data")
    train.add_argument("--data", help="file or folder with supported training data")
    train.add_argument("--epochs", type=int, default=1, help="number of replay passes")
    train.add_argument("--micro-hash-bits", type=int, choices=[16, 32], default=16, help="micro rank feature hash width")
    train.set_defaults(func=cmd_train)

    chat = sub.add_parser("chat", aliases=["c"], help="open an interactive chat shell")
    learn_group = chat.add_mutually_exclusive_group()
    learn_group.add_argument("--learn", dest="learn", action="store_true", default=True, help="learn from chat turns")
    learn_group.add_argument("--no-learn", dest="learn", action="store_false", help="do not learn from chat turns")
    chat.add_argument("--max-new-tokens", type=int, default=24)
    chat.set_defaults(func=cmd_chat)

    ask = sub.add_parser("ask", aliases=["q"], help="rank choices using brain memory resonance")
    ask.add_argument("question")
    ask.add_argument("choice_text", nargs="?", help="comma-separated choices")
    ask.add_argument("--choices", help="comma-separated choices")
    ask.add_argument("--mode", choices=sorted(ENERGY_MODES), default="event", help="energy scoring mode")
    ask.add_argument("--engine", choices=["auto", "micro", "full"], default="auto", help="inference engine")
    ask.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="micro inference device")
    ask.add_argument("--explain", action="store_true", help="print free-energy breakdown")
    ask.set_defaults(func=cmd_ask)

    ask_batch = sub.add_parser("ask-batch", aliases=["batch", "b"], help="rank many questions after loading the brain once")
    ask_batch.add_argument("input_path", nargs="?", help=".jsonl, .json, .csv, or .tsv with question and choices")
    ask_batch.add_argument("--input", help=".jsonl, .json, .csv, or .tsv with question and choices")
    ask_batch.add_argument("--output", help="optional JSONL output path")
    ask_batch.add_argument("--format", choices=["auto", "jsonl", "tsv"], default="auto", help="input format")
    ask_batch.add_argument("--mode", choices=sorted(ENERGY_MODES), default="event", help="energy scoring mode")
    ask_batch.add_argument("--engine", choices=["auto", "micro", "full"], default="auto", help="inference engine")
    ask_batch.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="micro inference device")
    ask_batch.add_argument("--quiet", action="store_true", help="do not print each JSON result")
    ask_batch.set_defaults(func=cmd_ask_batch)

    ask_stdin = sub.add_parser("ask-stdin", help="stream questions on stdin after loading the brain once")
    ask_stdin.add_argument("--format", choices=["jsonl", "tsv"], default="jsonl", help="jsonl or question<TAB>comma choices")
    ask_stdin.add_argument("--mode", choices=sorted(ENERGY_MODES), default="event", help="energy scoring mode")
    ask_stdin.add_argument("--engine", choices=["auto", "micro", "full"], default="auto", help="inference engine")
    ask_stdin.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="micro inference device")
    ask_stdin.add_argument("--scores", action="store_true", help="include scores in JSON output")
    ask_stdin.set_defaults(func=cmd_ask_stdin)

    recall = sub.add_parser("recall", help="retrieve learned events related to a query")
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=5)
    recall.set_defaults(func=cmd_recall)

    status = sub.add_parser("status", help="show brain files and memory stats")
    status.set_defaults(func=cmd_status)

    inspect = sub.add_parser("inspect", help="list learned events by source, modality, or text")
    inspect.add_argument("--source", help="source substring to match")
    inspect.add_argument("--modality", choices=["text", "image", "audio", "bytes"], help="event modality")
    inspect.add_argument("--contains", help="substring in source, preview, or features")
    inspect.add_argument("--limit", type=int, default=20)
    inspect.set_defaults(func=cmd_inspect)

    forget = sub.add_parser("forget", help="delete specific learned events and rebuild indexes")
    forget.add_argument("--event-id", action="append", help="event id to delete; can be repeated")
    forget.add_argument("--source", help="delete events whose source contains this text")
    forget.add_argument("--modality", choices=["text", "image", "audio", "bytes"], help="delete events by modality")
    forget.add_argument("--contains", help="delete events matching source, preview, or features")
    forget.add_argument("--all", action="store_true", help="allow deleting every event when no filter is supplied")
    forget.add_argument("--dry-run", action="store_true", help="show matching events without deleting")
    forget.add_argument("--yes", action="store_true", help="confirm deletion")
    forget.add_argument("--limit", type=int, default=50, help="preview limit before deletion")
    forget.set_defaults(func=cmd_forget)

    explain = sub.add_parser("explain", help="show event and feature evidence for a ranked answer")
    explain.add_argument("question")
    explain.add_argument("--choices", required=True, help="comma-separated choices")
    explain.add_argument("--mode", choices=sorted(ENERGY_MODES), default="event", help="energy scoring mode")
    explain.add_argument("--top-events", type=int, default=5, help="number of supporting events to show")
    explain.set_defaults(func=cmd_explain)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    normalized = normalize_default_argv(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(normalized)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
