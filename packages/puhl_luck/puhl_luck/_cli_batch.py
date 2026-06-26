from __future__ import annotations

from ._cli_common import *
from ._cli_store import open_brain, open_brain_for_rank

def _row_choices(row: dict[str, Any]) -> List[str]:
    choices = row.get("choices", row.get("options", row.get("candidates", [])))
    if isinstance(choices, str):
        return [c.strip() for c in choices.split(",") if c.strip()]
    if isinstance(choices, list):
        return [str(c).strip() for c in choices if str(c).strip()]
    return []


def iter_ask_rows(path: Path, input_format: str = "jsonl") -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if input_format == "auto":
        input_format = "tsv" if suffix == ".tsv" else "jsonl"
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
    input_value = args.input_path or args.input
    if not input_value:
        raise SystemExit("ask-batch needs an input file, for example: puhl-luck b ask.tsv")
    rows = iter_ask_rows(Path(input_value), args.format)
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
