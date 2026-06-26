from __future__ import annotations

from ._cli_common import *
from ._cli_store import open_brain, open_brain_for_rank

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
