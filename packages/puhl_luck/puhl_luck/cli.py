from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .brain_memory import ENERGY_MODES, TEXT_SUFFIXES, BrainMemory, MicroRankModel


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_BRAIN_DIR = Path("brain_data")
DEFAULT_DATA_DIR = Path("data")
TRAIN_HINT = "put files in data/ and run: puhl-luck train"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value


def read_text_rows(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [line.strip().lstrip("\ufeff") for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines() if line.strip()]
    if suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                text = " ".join(str(cell).strip() for cell in row if str(cell).strip())
                if text:
                    rows.append(text)
        return rows
    if suffix == ".jsonl":
        rows = []
        preferred = ("text", "content", "question", "prompt", "input")
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    rows.append(line)
                    continue
                if isinstance(obj, dict):
                    parts = [str(obj[k]).strip() for k in preferred if k in obj and str(obj[k]).strip()]
                    rows.append(" ".join(parts) if parts else json.dumps(obj, ensure_ascii=False))
                else:
                    rows.append(str(obj))
        return rows
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        if isinstance(obj, list):
            return [json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x for x in obj]
        if isinstance(obj, dict):
            return [json.dumps(obj, ensure_ascii=False)]
    return []


def resolve_training_data_path(data_arg: Optional[str]) -> Path:
    if data_arg:
        return Path(data_arg)
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR
    raise SystemExit(f"no training data path provided and default folder does not exist: {DEFAULT_DATA_DIR}\n{TRAIN_HINT}")


def iter_training_inputs(data_path: Path) -> Iterable[tuple[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"data path not found: {data_path}")
    files = sorted(p for p in data_path.rglob("*") if p.is_file()) if data_path.is_dir() else [data_path]
    for path in files:
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            for text in read_text_rows(path):
                if text.strip():
                    yield "text", text
        else:
            yield "file", path


class BrainStore:
    def __init__(self, brain_dir: Path, window_size: int = 12):
        self.brain_dir = brain_dir
        self.window_size = int(window_size)
        self.memory = BrainMemory(window_size=window_size)
        self.meta = {
            "created_at": time.time(),
            "updated_at": None,
            "window_size": window_size,
            "interaction_learning_default": True,
            "architecture": "unified_event_memory",
        }

    @property
    def memory_path(self) -> Path:
        return self.brain_dir / "brain_memory.pkl"

    @property
    def micro_path(self) -> Path:
        return self.brain_dir / "brain_rank_micro.pmr"

    @property
    def meta_path(self) -> Path:
        return self.brain_dir / "brain_meta.json"

    @property
    def log_path(self) -> Path:
        return self.brain_dir / "interaction_log.jsonl"

    def load(self) -> bool:
        if not self.memory_path.exists():
            return False
        self.memory = BrainMemory.load(self.memory_path)
        if self.meta_path.exists():
            self.meta.update(json.loads(self.meta_path.read_text(encoding="utf-8")))
        return True

    def current_micro_hash_bits(self) -> int:
        try:
            value = int(self.meta.get("files", {}).get("brain_rank_micro_hash_bits", 16))
        except (TypeError, ValueError):
            value = 16
        return 32 if value == 32 else 16

    def save(self, micro_hash_bits: Optional[int] = None) -> None:
        selected_hash_bits = self.current_micro_hash_bits() if micro_hash_bits is None else (32 if int(micro_hash_bits) == 32 else 16)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self.memory.prune()
        self.memory.save(self.memory_path)
        self.memory.save_rank_micro_only(self.micro_path, hash_bits=selected_hash_bits)
        self.meta["updated_at"] = time.time()
        self.meta["stats"] = self.memory.stats()
        self.meta["files"] = {
            "brain_memory_bytes": self.memory_path.stat().st_size if self.memory_path.exists() else 0,
            "brain_rank_micro_bytes": self.micro_path.stat().st_size if self.micro_path.exists() else 0,
            "brain_rank_micro_hash_bits": selected_hash_bits,
        }
        self.meta_path.write_text(json.dumps(json_safe(self.meta), ensure_ascii=False, indent=2), encoding="utf-8")

    def learn_inputs(self, inputs: Iterable[tuple[str, Any]], epochs: int = 1, verbose: bool = True) -> int:
        rows = list(inputs)
        learned = 0
        for epoch in range(max(1, int(epochs))):
            for kind, value in rows:
                if kind == "text":
                    self.memory.expose_text(str(value), source="training_text")
                else:
                    self.memory.expose_file(Path(value))
                learned += 1
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}: learned {len(rows)} events")
        return learned

    def learn_data_path(self, data_path: Path, epochs: int = 1, micro_hash_bits: int = 16, verbose: bool = True) -> int:
        try:
            inputs = list(iter_training_inputs(data_path))
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        if not inputs:
            raise SystemExit(f"no supported training input found in {data_path}")
        learned = self.learn_inputs(inputs, epochs=epochs, verbose=verbose)
        self.save(micro_hash_bits=micro_hash_bits)
        return learned

    def learn_interaction(self, user_text: str, assistant_text: str) -> None:
        self.memory.expose_text(f"user: {user_text}", source="chat:user")
        self.memory.expose_text(f"assistant: {assistant_text}", source="chat:assistant")
        self.memory.expose_text(f"user: {user_text}\nassistant: {assistant_text}", source="chat:turn")
        self.log_interaction(user_text, assistant_text, learned=True)

    def log_interaction(self, user_text: str, assistant_text: str, learned: bool) -> None:
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        row = {"time": time.time(), "user": user_text, "assistant": assistant_text, "learned": learned}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def answer(self, prompt: str, max_new_tokens: int = 24) -> str:
        return self.memory.answer(prompt, max_new_tokens=max_new_tokens)

    def stats(self) -> dict:
        return {
            "brain_dir": str(self.brain_dir),
            "memory": self.memory.stats(),
            "meta": self.meta,
            "files": {
                "brain_memory": str(self.memory_path),
                "brain_rank_micro": str(self.micro_path),
                "meta": str(self.meta_path),
                "interaction_log": str(self.log_path),
            },
        }

    def micro_ranker(self) -> Optional[MicroRankModel]:
        if not self.micro_path.exists():
            return None
        return MicroRankModel.load(self.micro_path)

    def has_full_memory(self) -> bool:
        return self.memory_path.exists()

    def has_micro_memory(self) -> bool:
        return self.micro_path.exists()

    def require_trained(self) -> None:
        if not self.has_full_memory() and not self.has_micro_memory():
            raise SystemExit(
                f"no trained brain found in {self.brain_dir}\n"
                f"run: puhl-luck --brain-dir {self.brain_dir} train --data data"
            )


def open_brain(args, load: bool = True) -> BrainStore:
    brain = BrainStore(Path(args.brain_dir), window_size=args.window_size)
    if load:
        brain.load()
    return brain


def open_brain_for_rank(args, explain: bool = False) -> tuple[BrainStore, Optional[MicroRankModel]]:
    wants_micro = args.engine in {"auto", "micro"} and args.mode == "event" and not explain
    brain = open_brain(args, load=not wants_micro)
    ranker = brain.micro_ranker() if wants_micro else None
    if ranker is not None:
        return brain, ranker
    if args.engine == "micro":
        raise SystemExit(f"micro rank file not found or unavailable: {brain.micro_path}")
    if wants_micro:
        brain.load()
    brain.require_trained()
    return brain, None


def cmd_train(args) -> None:
    brain = open_brain(args)
    data_path = resolve_training_data_path(args.data)
    learned = brain.learn_data_path(data_path, epochs=args.epochs, micro_hash_bits=args.micro_hash_bits)
    print(f"data: {data_path}")
    print(f"trained: {learned} events")
    print(f"saved full: {brain.memory_path}")
    print(f"saved micro: {brain.micro_path}")
    print(f"micro hash bits: {32 if int(args.micro_hash_bits) == 32 else 16}")
    print(f"ask: puhl-luck --brain-dir {brain.brain_dir} ask \"your question\" --choices \"A,B\"")


def chat_help() -> str:
    return "\n".join([
        "commands:",
        "  /learn on              enable learning from chat turns",
        "  /learn off             disable learning from chat turns",
        "  /quiz [topic]          make a multiple-choice question from memory",
        "  /rank question | A,B   rank choices inside chat",
        "  /save                  save the current brain",
        "  /stats                 print memory stats",
        "  /exit                  quit",
    ])


def preview_text(value: str, limit: int = 96) -> str:
    text = " ".join(str(value).replace("\ufeff", "").split())
    return text[:limit] if len(text) <= limit else text[: limit - 3] + "..."


def build_quiz(brain: BrainStore, topic: str = "") -> Optional[dict[str, Any]]:
    rows = brain.memory.recall(topic, limit=6) if topic else []
    if not rows:
        rows = []
        for eid in reversed(list(brain.memory.events.keys())):
            rec = brain.memory.events[eid]
            if rec.preview:
                rows.append({"event_id": eid, "preview": rec.preview})
            if len(rows) >= 6:
                break
    if not rows:
        return None
    correct_row = rows[0]
    correct = preview_text(correct_row.get("preview", ""))
    if not correct:
        return None
    choices = [correct]
    seen = {correct}
    for row in rows[1:]:
        text = preview_text(row.get("preview", ""))
        if text and text not in seen:
            choices.append(text)
            seen.add(text)
        if len(choices) >= 4:
            break
    if len(choices) < 2:
        for eid, rec in brain.memory.events.items():
            text = preview_text(rec.preview)
            if text and text not in seen:
                choices.append(text)
                seen.add(text)
            if len(choices) >= 4:
                break
    if len(choices) < 2:
        return None
    rng = random.Random(correct)
    rng.shuffle(choices)
    answer = choices.index(correct)
    subject = topic.strip() if topic.strip() else "recent memory"
    return {
        "question": f"Which learned memory is most related to: {subject}?",
        "choices": choices,
        "answer": answer,
    }


def print_quiz(quiz: dict[str, Any]) -> None:
    print(f"quiz> {quiz['question']}")
    for idx, choice in enumerate(quiz["choices"], start=1):
        print(f"  {idx}. {choice}")
    print("answer with the number.")


def parse_rank_command(text: str) -> tuple[str, List[str]]:
    body = text.split(" ", 1)[1].strip() if " " in text else ""
    if "|" not in body:
        raise ValueError("usage: /rank question | choice1,choice2")
    question, choices_text = body.split("|", 1)
    choices = [choice.strip() for choice in choices_text.split(",") if choice.strip()]
    if not question.strip() or not choices:
        raise ValueError("usage: /rank question | choice1,choice2")
    return question.strip(), choices


def run_chat_loop(brain: BrainStore, args, learn_enabled: bool) -> None:
    learn_enabled = args.learn
    print(f"brain: {brain.brain_dir}")
    print(f"interaction learning: {'on' if learn_enabled else 'off'}")
    print(chat_help())
    pending_quiz: Optional[dict[str, Any]] = None
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        low = user_text.lower()
        if low in {"/exit", "/quit", "exit", "quit"}:
            break
        if low in {"/help", "help"}:
            print(chat_help())
            continue
        if low == "/learn on":
            learn_enabled = True
            brain.meta["interaction_learning_default"] = True
            print("learning is on")
            continue
        if low == "/learn off":
            learn_enabled = False
            brain.meta["interaction_learning_default"] = False
            print("learning is off")
            continue
        if low == "/save":
            brain.save()
            print(f"saved: {brain.memory_path}")
            continue
        if low == "/stats":
            print(json.dumps(json_safe(brain.stats()), ensure_ascii=False, indent=2))
            continue
        if low.startswith("/quiz"):
            topic = user_text.split(" ", 1)[1].strip() if " " in user_text else ""
            quiz = build_quiz(brain, topic)
            if quiz is None:
                print("quiz> not enough learned memory yet.")
                continue
            pending_quiz = quiz
            print_quiz(quiz)
            if learn_enabled:
                brain.memory.expose_text(f"quiz requested topic: {topic or 'recent memory'}", source="chat:quiz")
                brain.save()
            continue
        if low.startswith("/rank"):
            try:
                question, choices = parse_rank_command(user_text)
                pred, scores = brain.memory.rank(question, choices)
            except ValueError as exc:
                print(f"rank> {exc}")
                continue
            print(f"rank> answer: {choices[pred]}")
            for idx, (choice, score) in enumerate(zip(choices, scores), start=1):
                marker = "*" if idx - 1 == pred else " "
                print(f"{marker} {idx}. {choice} score={score:.4f}")
            if learn_enabled:
                brain.memory.expose_text(f"rank interaction question: {question} choices: {', '.join(choices)} answer: {choices[pred]}", source="chat:rank")
                brain.save()
            continue
        if pending_quiz is not None:
            answer_text = user_text.strip()
            correct_idx = int(pending_quiz["answer"])
            correct_text = pending_quiz["choices"][correct_idx]
            selected: Optional[int] = None
            if answer_text.isdigit():
                selected = int(answer_text) - 1
            else:
                for idx, choice in enumerate(pending_quiz["choices"]):
                    if answer_text.lower() == choice.lower():
                        selected = idx
                        break
            if selected == correct_idx:
                print("quiz> correct.")
            else:
                print(f"quiz> wrong. answer: {correct_idx + 1}. {correct_text}")
            if learn_enabled:
                brain.memory.expose_text(
                    f"quiz answer question: {pending_quiz['question']} user: {user_text} correct: {correct_text}",
                    source="chat:quiz_answer",
                )
                brain.save()
            pending_quiz = None
            continue

        assistant_text = brain.answer(user_text, max_new_tokens=args.max_new_tokens)
        print(f"brain> {assistant_text}")
        if learn_enabled:
            brain.learn_interaction(user_text, assistant_text)
            brain.save()
        else:
            brain.log_interaction(user_text, assistant_text, learned=False)
    brain.save()
    print(f"saved: {brain.memory_path}")


def cmd_chat(args) -> None:
    brain = open_brain(args)
    run_chat_loop(brain, args, learn_enabled=args.learn)


def cmd_start(args) -> None:
    brain = open_brain(args)
    data_path = Path(args.data) if args.data else DEFAULT_DATA_DIR
    if data_path.exists():
        learned = brain.learn_data_path(data_path, epochs=args.epochs, micro_hash_bits=args.micro_hash_bits)
        print(f"auto-trained: {learned} events from {data_path}")
        print(f"saved micro: {brain.micro_path}")
    else:
        print(f"no data folder found: {data_path}")
        print("starting chat; interaction learning can still create the brain.")
    run_chat_loop(brain, args, learn_enabled=args.learn)


def cmd_ask(args) -> None:
    brain, ranker = open_brain_for_rank(args, explain=args.explain)
    choices = [c.strip() for c in args.choices.split(",") if c.strip()] if args.choices else []
    if not choices:
        choices = [c.strip() for c in input("choices comma-separated> ").split(",") if c.strip()]
    if not choices:
        raise SystemExit("ask needs choices")
    if ranker is not None:
        try:
            pred, scores = ranker.rank(args.question, choices, device=args.device)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        breakdown = []
    else:
        pred, scores = brain.memory.rank(args.question, choices, mode=args.mode)
        breakdown = brain.memory.rank_energy_breakdown(args.question, choices) if args.explain else []
    print(f"question: {args.question}")
    for i, (choice, score) in enumerate(zip(choices, scores), start=1):
        marker = "*" if i - 1 == pred else " "
        if args.explain:
            row = breakdown[i - 1]
            print(
                f"{marker} {i}. {choice} score={score:.4f} "
                f"free_energy={row['free_energy']:.4f} evidence={row['score']:.4f} "
                f"overlap={row['weighted_overlap']:.4f} event={row['event_support']:.4f} "
                f"align={row['alignment']:.4f} conflict={row['conflict']:.4f}"
            )
        else:
            print(f"{marker} {i}. {choice} score={score:.4f}")
    print(f"answer: {choices[pred]}")


def _row_choices(row: dict[str, Any]) -> List[str]:
    choices = row.get("choices", row.get("options", row.get("candidates", [])))
    if isinstance(choices, str):
        return [c.strip() for c in choices.split(",") if c.strip()]
    if isinstance(choices, list):
        return [str(c).strip() for c in choices if str(c).strip()]
    return []


def iter_ask_rows(path: Path, input_format: str = "jsonl") -> Iterable[dict[str, Any]]:
    if input_format == "tsv":
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    yield {"question": parts[0], "choices": [c.strip() for c in parts[1].split(",") if c.strip()]}
        return
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        if isinstance(obj, list):
            for row in obj:
                if isinstance(row, dict):
                    yield row
        elif isinstance(obj, dict):
            rows = obj.get("rows", obj.get("data", []))
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        yield row
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        return
    raise SystemExit("ask-batch input must be .jsonl, .json, or .csv")


def cmd_ask_batch(args) -> None:
    brain, ranker = open_brain_for_rank(args)
    rows = iter_ask_rows(Path(args.input), args.format)
    out_f = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
    try:
        for idx, row in enumerate(rows):
            question = str(row.get("question", row.get("prompt", row.get("input", ""))))
            choices = _row_choices(row)
            if not question or not choices:
                result = {"index": idx, "error": "missing question or choices"}
            else:
                if ranker is not None:
                    try:
                        if args.quiet and out_f is None:
                            ranker.predict(question, choices, device=args.device)
                            continue
                        pred, scores = ranker.rank(question, choices, device=args.device)
                    except RuntimeError as exc:
                        raise SystemExit(str(exc)) from exc
                else:
                    pred, scores = brain.memory.rank(question, choices, mode=args.mode)
                if args.quiet and out_f is None:
                    continue
                result = {
                    "index": idx,
                    "prediction": pred,
                    "answer": choices[pred],
                    "scores": scores,
                }
            line = json.dumps(json_safe(result), ensure_ascii=False)
            if out_f is not None:
                out_f.write(line + "\n")
            if not args.quiet:
                print(line)
    finally:
        if out_f is not None:
            out_f.close()


def cmd_ask_stdin(args) -> None:
    brain, ranker = open_brain_for_rank(args)
    for idx, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue
        try:
            if args.format == "tsv":
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    raise ValueError("tsv input must be: question<TAB>choice1,choice2")
                question = parts[0]
                choices = [c.strip() for c in parts[1].split(",") if c.strip()]
            else:
                row = json.loads(line)
                question = str(row.get("question", row.get("prompt", row.get("input", ""))))
                choices = _row_choices(row)
            if not question or not choices:
                result = {"index": idx, "error": "missing question or choices"}
            else:
                if ranker is not None and not args.scores:
                    try:
                        pred = ranker.predict(question, choices, device=args.device)
                    except RuntimeError as exc:
                        raise SystemExit(str(exc)) from exc
                    scores = []
                elif ranker is not None:
                    try:
                        pred, scores = ranker.rank(question, choices, device=args.device)
                    except RuntimeError as exc:
                        raise SystemExit(str(exc)) from exc
                else:
                    pred, scores = brain.memory.rank(question, choices, mode=args.mode)
                if args.format == "tsv" and not args.scores:
                    print(str(pred))
                    continue
                result = {"index": idx, "prediction": pred, "answer": choices[pred]}
                if args.scores:
                    result["scores"] = scores
        except Exception as exc:
            result = {"index": idx, "error": str(exc)}
        print(json.dumps(json_safe(result), ensure_ascii=False))


def cmd_recall(args) -> None:
    brain = open_brain(args)
    rows = brain.memory.recall(args.query, limit=args.limit)
    print(json.dumps(json_safe(rows), ensure_ascii=False, indent=2))


def cmd_status(args) -> None:
    brain = open_brain(args)
    print(json.dumps(json_safe(brain.stats()), ensure_ascii=False, indent=2))


def cmd_inspect(args) -> None:
    brain = open_brain(args)
    rows = brain.memory.inspect_events(
        source=args.source,
        modality=args.modality,
        contains=args.contains,
        limit=args.limit,
    )
    print(json.dumps(json_safe(rows), ensure_ascii=False, indent=2))


def cmd_forget(args) -> None:
    brain = open_brain(args)
    event_ids = args.event_id or []
    if not (event_ids or args.source or args.modality or args.contains or args.all):
        raise SystemExit("forget needs --event-id, --source, --modality, --contains, or --all")
    matched = brain.memory.inspect_events(
        source=args.source,
        modality=args.modality,
        contains=args.contains,
        limit=max(1, args.limit),
    )
    if event_ids:
        wanted = set(event_ids)
        for eid in event_ids:
            rec = brain.memory.events.get(eid)
            if rec is not None and not any(row["event_id"] == eid for row in matched):
                matched.append({
                    "event_id": eid,
                    "source": rec.source,
                    "modality": rec.modality,
                    "label": rec.label,
                    "preview": rec.preview,
                    "feature_count": len(rec.features),
                    "created_at": rec.created_at,
                    "last_accessed_at": rec.last_accessed_at,
                })
        matched = [row for row in matched if row["event_id"] in wanted or not event_ids]
    if args.dry_run:
        print(json.dumps(json_safe({"matched": len(matched), "events": matched}), ensure_ascii=False, indent=2))
        return
    if not args.yes:
        print(json.dumps(json_safe({"matched": len(matched), "events": matched}), ensure_ascii=False, indent=2))
        raise SystemExit("add --yes to delete these learned events")
    result = brain.memory.forget_events(
        event_ids=event_ids,
        source=args.source,
        modality=args.modality,
        contains=args.contains,
    )
    brain.save()
    result["saved"] = {
        "brain_memory": str(brain.memory_path),
        "brain_rank_micro": str(brain.micro_path),
    }
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


def cmd_explain(args) -> None:
    brain = open_brain(args)
    choices = [c.strip() for c in args.choices.split(",") if c.strip()] if args.choices else []
    if not choices:
        raise SystemExit("explain needs --choices")
    result = brain.memory.explain_rank(args.question, choices, mode=args.mode, top_events=args.top_events)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="puhl-luck", description="Train and chat with a local unified brain memory.")
    parser.add_argument("--brain-dir", default=str(DEFAULT_BRAIN_DIR), help="folder that stores the learned brain file")
    parser.add_argument("--window-size", type=int, default=12, help="event co-activation window")
    sub = parser.add_subparsers(dest="command")

    start = sub.add_parser("start", help="auto-train data folder and open chat")
    start.add_argument("--data", help="file or folder to auto-train before chat; default is data")
    start.add_argument("--epochs", type=int, default=1, help="number of replay passes")
    start.add_argument("--micro-hash-bits", type=int, choices=[16, 32], default=16, help="micro rank feature hash width")
    start_learn_group = start.add_mutually_exclusive_group()
    start_learn_group.add_argument("--learn", dest="learn", action="store_true", default=True, help="learn from chat turns")
    start_learn_group.add_argument("--no-learn", dest="learn", action="store_false", help="do not learn from chat turns")
    start.add_argument("--max-new-tokens", type=int, default=24)
    start.set_defaults(func=cmd_start)

    train = sub.add_parser("train", aliases=["learn"], help="learn text, image, audio, and byte events")
    train.add_argument("--data", help="file or folder with supported training data")
    train.add_argument("--epochs", type=int, default=1, help="number of replay passes")
    train.add_argument("--micro-hash-bits", type=int, choices=[16, 32], default=16, help="micro rank feature hash width")
    train.set_defaults(func=cmd_train)

    chat = sub.add_parser("chat", help="open an interactive chat shell")
    learn_group = chat.add_mutually_exclusive_group()
    learn_group.add_argument("--learn", dest="learn", action="store_true", default=True, help="learn from chat turns")
    learn_group.add_argument("--no-learn", dest="learn", action="store_false", help="do not learn from chat turns")
    chat.add_argument("--max-new-tokens", type=int, default=24)
    chat.set_defaults(func=cmd_chat)

    ask = sub.add_parser("ask", help="rank choices using brain memory resonance")
    ask.add_argument("question")
    ask.add_argument("--choices", help="comma-separated choices")
    ask.add_argument("--mode", choices=sorted(ENERGY_MODES), default="event", help="energy scoring mode")
    ask.add_argument("--engine", choices=["auto", "micro", "full"], default="auto", help="inference engine")
    ask.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="micro inference device")
    ask.add_argument("--explain", action="store_true", help="print free-energy breakdown")
    ask.set_defaults(func=cmd_ask)

    ask_batch = sub.add_parser("ask-batch", help="rank many questions after loading the brain once")
    ask_batch.add_argument("--input", required=True, help=".jsonl, .json, or .csv with question and choices")
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
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
