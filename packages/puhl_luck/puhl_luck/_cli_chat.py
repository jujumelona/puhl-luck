from __future__ import annotations

from ._cli_common import *
from ._cli_store import BrainStore, open_brain

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
    data_value = args.data_path or args.data
    data_path = Path(data_value) if data_value else DEFAULT_DATA_DIR
    if data_path.exists():
        learned = brain.learn_data_path(data_path, epochs=args.epochs, micro_hash_bits=args.micro_hash_bits)
        print(f"auto-trained: {learned} events from {data_path}")
        print(f"saved micro: {brain.micro_path}")
    else:
        print(f"no data folder found: {data_path}")
        print("starting chat; interaction learning can still create the brain.")
    run_chat_loop(brain, args, learn_enabled=args.learn)
