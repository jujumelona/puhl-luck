from __future__ import annotations

from ._cli_common import *

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
