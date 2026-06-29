from __future__ import annotations

from ._cli_common import *
from ._cli_store import open_brain, open_brain_for_rank

def cmd_train(args) -> None:
    brain = open_brain(args)
    data_path = resolve_training_data_path(args.data_path or args.data)
    learned = brain.learn_data_path(data_path, epochs=args.epochs, micro_hash_bits=args.micro_hash_bits)
    print(f"data: {data_path}")
    print(f"trained: {learned} events")
    print(f"saved full: {brain.memory_path}")
    print(f"saved micro: {brain.micro_path}")
    print(f"micro hash bits: {32 if int(args.micro_hash_bits) == 32 else 16}")
    print(f"ask: puhl-luck q \"your question\" \"A,B\"")

def cmd_ask(args) -> None:
    brain, ranker = open_brain_for_rank(args, explain=args.explain)
    choices_value = args.choice_text or args.choices
    choices = [c.strip() for c in choices_value.split(",") if c.strip()] if choices_value else []
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

def cmd_explain(args) -> None:
    brain = open_brain(args)
    choices = [c.strip() for c in args.choices.split(",") if c.strip()] if args.choices else []
    if not choices:
        raise SystemExit("explain needs --choices")
    result = brain.memory.explain_rank(args.question, choices, mode=args.mode, top_events=args.top_events)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
