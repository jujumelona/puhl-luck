from pathlib import Path
import time
import zlib

import numpy as np

from puhl_luck import BrainMemory, MicroRankModel
from puhl_luck.brain_memory import bundle_hv, hv_similarity, text_feature_list, tokenize, write_varuint


def test_text_memory_learns_and_ranks():
    mem = BrainMemory(window_size=8)
    mem.expose_text("python is used for machine learning", source="unit")
    mem.expose_text("css styles web pages", source="unit")

    pred, scores = mem.rank("machine learning language", ["CSS", "Python"])

    assert pred == 1
    assert scores[1] > scores[0]
    assert mem.stats()["events"] == 2
    assert mem.stats()["features"] > 0


def test_memory_card_id_stays_as_anchor_token():
    assert tokenize("lumora_001 winter echoes")[:3] == ["lumora_001", "winter", "echoes"]
    features = text_feature_list("lumora_001 winter echoes")
    assert "text:id:lumora_001" in features


def test_anchor_id_beats_common_char_features():
    mem = BrainMemory(window_size=6)
    mem.expose_text("memory card lumora_001 winter echoes blue lantern", source="cards")
    mem.expose_text("memory card naruvol_003 winter echoes red tower", source="cards")
    for i in range(20):
        mem.expose_text(f"memory card common_{i:03d} winter echoes card memory archive", source="noise")

    pred, scores = mem.rank("lumora_001 winter echoes", ["naruvol_003", "lumora_001"], mode="event")
    state = mem._rank_query_state("lumora_001 winter echoes")
    top_event = state["event_scores"].most_common(1)[0][0]

    assert pred == 1
    assert scores[1] > scores[0]
    assert "lumora_001" in mem.events[top_event].preview


def test_continuous_hopfield_recall_finds_anchor_event():
    mem = BrainMemory(window_size=6)
    mem.expose_text("memory card lumora_001 winter echoes blue lantern", source="cards")
    mem.expose_text("memory card naruvol_003 winter echoes red tower", source="cards")

    query_features = mem.features_for_query("lumora_001 winter echoes")
    scores = mem.hopfield_recall_feature_continuous(query_features, iterations=2, top_k=2)
    top_event = max(scores.items(), key=lambda item: item[1])[0]

    assert "lumora_001" in mem.events[top_event].preview


def test_repeated_same_fact_aggregates_one_event():
    mem = BrainMemory()
    first = mem.expose_text("memory card lumora_001 winter echoes blue lantern", source="a")
    second = mem.expose_text("memory card lumora_001 winter echoes blue lantern", source="b")
    fid = mem.feature_to_id["text:id:lumora_001"]

    assert first == second
    assert mem.stats()["events"] == 1
    assert mem.feature_to_events[fid][first] == 2
    assert "a" in mem.events[first].source
    assert "b" in mem.events[first].source


def test_file_modalities_share_one_memory(tmp_path: Path):
    img = tmp_path / "sample.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (16).to_bytes(4, "big") + (8).to_bytes(4, "big") + b"payload")
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    mem = BrainMemory(window_size=8)
    mem.expose_file(img)
    mem.expose_file(wav)

    stats = mem.stats()
    assert stats["modalities"]["image"] == 1
    assert stats["modalities"]["audio"] == 1
    assert stats["events"] == 2


def test_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "brain_memory.pkl"
    mem = BrainMemory()
    mem.expose_text("memory survives save load", source="unit")
    mem.save(path)

    loaded = BrainMemory.load(path)

    assert loaded.stats()["events"] == 1
    assert loaded.recall("save load", limit=1)


def test_forget_source_removes_addressed_memory():
    mem = BrainMemory()
    keep_id = mem.expose_text("python machine learning model", source="good_source")
    drop_id = mem.expose_text("css browser layout style", source="bad_source")

    result = mem.forget_events(source="bad_source")

    assert result["removed"] == 1
    assert keep_id in mem.events
    assert drop_id not in mem.events
    assert drop_id not in mem.event_hv
    assert "text:tok:css" not in mem.feature_freq
    assert all(drop_id not in rows for rows in mem.feature_to_events.values())
    assert not mem.inspect_events(source="bad_source")


def test_explain_rank_reports_supporting_events():
    mem = BrainMemory()
    mem.expose_text("python machine learning model", source="python_doc")
    mem.expose_text("css browser layout style", source="css_doc")

    result = mem.explain_rank("machine learning", ["CSS", "Python"], mode="event")

    assert result["prediction"] == 1
    assert result["answer"] == "Python"
    assert any(row["source"] == "python_doc" for row in result["events"])


def test_answer_uses_recalled_sequence_trace():
    mem = BrainMemory()
    mem.expose_text("red blue green apple", source="unit")
    mem.expose_text("red blue yellow banana", source="unit")

    answer = mem.answer("red blue green", max_new_tokens=1)

    assert "apple" in answer.lower()


def test_answer_composes_from_learned_text_memory():
    mem = BrainMemory()
    mem.expose_text("moon city silver towers quiet market", source="story")
    mem.expose_text("desert engine solar caravan repair crew", source="story")

    answer = mem.answer("write moon city scene", max_new_tokens=16)

    assert "moon" in answer
    assert "city" in answer


def test_graph_decoder_creates_sequence_from_memory_graph():
    mem = BrainMemory()
    mem.expose_text("moon city silver towers quiet market", source="story")
    mem.expose_text("moon caravan crosses desert engine lights", source="story")

    answer = mem.graph_decode_text("moon caravan", ["moon", "caravan"], max_new_tokens=8)

    assert answer.startswith("moon caravan")
    assert len(answer.split()) > 2
    assert "crosses" in answer


def test_generation_backs_off_to_shorter_suffix():
    mem = BrainMemory()
    mem.expose_text("alpha beta gamma delta", source="unit")

    answer = mem.graph_decode_text("unknown alpha beta", ["unknown", "alpha", "beta"], max_new_tokens=2)

    assert answer == "unknown alpha beta gamma delta"


def test_generation_returns_empty_without_order_context():
    mem = BrainMemory()
    mem.expose_text("alpha beta gamma delta", source="unit")

    answer = mem.graph_decode_text("never seen prefix", ["never", "seen", "prefix"], max_new_tokens=2)

    assert answer == ""


def test_memory_energy_decoder_uses_semantic_candidates():
    mem = BrainMemory()
    mem.expose_text("mist valley opens obsidian gate under rain", source="unit")
    mem.expose_text("silent archive keeps amber key near river", source="unit")

    answer = mem.memory_energy_decode_text("mist archive", ["mist", "archive"], max_new_tokens=4)

    assert answer
    assert any(token in answer for token in ("opens", "keeps", "obsidian", "amber"))


def test_memory_energy_decoder_has_global_fallback():
    mem = BrainMemory()
    mem.expose_text("mist valley opens obsidian gate under rain", source="unit")

    answer = mem.memory_energy_decode_text("purple comet", ["purple", "comet"], max_new_tokens=4)

    assert answer
    assert answer.startswith("purple comet ")


def test_answer_handles_file_input_through_same_memory(tmp_path: Path):
    img = tmp_path / "scene.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (32).to_bytes(4, "big") + (16).to_bytes(4, "big") + b"payload")
    mem = BrainMemory()
    mem.expose_file(img)

    answer = mem.answer(str(img), max_new_tokens=16)

    assert "image" in answer
    assert "bytes" in answer


def test_hdc_similarity_gives_nonzero_fallback():
    left = bundle_hv(["mod:text", "text:tok:python"])
    right = bundle_hv(["mod:text", "text:tok:ruby"])

    assert left.dtype == np.uint64
    assert hv_similarity(left, right) > 0.0


def test_edges_use_compact_integer_ids():
    mem = BrainMemory(window_size=4)
    mem.expose_text("alpha beta gamma", source="unit")

    assert mem.edges
    assert all(isinstance(edge, tuple) and all(isinstance(fid, int) for fid in edge) for edge in mem.edges.keys())


def test_hdc_resolution_is_dynamic():
    mem = BrainMemory()
    start_bits = mem.hdc_bits
    for i in range(40):
        mem.expose_text(f"unique token stream {i} alpha beta gamma", source="unit")

    assert mem.hdc_bits > start_bits


def test_hdc_growth_preserves_existing_prefix():
    mem = BrainMemory()
    mem.expose_text("alpha beta gamma", source="unit")
    event_id = next(iter(mem.event_hv))
    old_words = mem.hdc_words
    old_vec = mem.event_hv[event_id].copy()

    for i in range(60):
        mem.expose_text(f"new unique stream {i} delta epsilon zeta", source="unit")

    assert mem.event_hv[event_id].size >= old_words
    assert np.array_equal(mem.event_hv[event_id][:old_words], old_vec)


def test_hdc_bundle_prefix_is_stable_across_width_growth():
    features = ["mod:text", "text:tok:alpha", "text:tok:beta", "text:bi:alpha_beta"]

    narrow = bundle_hv(features, bits=2 * 64)
    wide = bundle_hv(features, bits=8 * 64)

    assert np.array_equal(narrow, wide[: narrow.size])


def test_hdc_can_recall_same_text_after_width_growth():
    mem = BrainMemory()
    event_id = mem.expose_text("alpha beta stable anchor", source="unit")
    old_words = mem.hdc_words
    old_vec = mem.event_hv[event_id].copy()

    for i in range(80):
        mem.expose_text(f"unrelated growth stream {i} delta epsilon zeta", source="unit")

    query_features, query_sequence, _ = mem.extract_text("alpha beta stable anchor")
    query_vec = mem.bundle_event(query_features, query_sequence)

    assert mem.hdc_words > old_words
    assert np.array_equal(query_vec[:old_words], old_vec)
    assert event_id in mem.hdc_candidates(query_vec)


def test_repeated_clusters_form_concept_nodes():
    mem = BrainMemory()
    for i in range(3):
        mem.expose_text(f"alpha beta gamma concept seed {i}", source="unit")

    assert mem.stats()["concepts"] > 0


def test_graph_edges_are_directional():
    mem = BrainMemory(window_size=4)
    mem.expose_text("aa bb", source="unit")
    alpha = mem.feature_to_id["text:tok:aa"]
    beta = mem.feature_to_id["text:tok:bb"]

    assert mem.edges[(alpha, beta)] > mem.edges[(beta, alpha)]


def test_rank_does_not_slow_linearly_with_memory_growth():
    small = BrainMemory(window_size=4)
    for i in range(25):
        small.expose_text(f"alpha beta target small {i}", source="unit")

    large = BrainMemory(window_size=4)
    for i in range(500):
        large.expose_text(f"noise stream unrelated {i} token {i % 17}", source="unit")
    for i in range(25):
        large.expose_text(f"alpha beta target large {i}", source="unit")

    choices = ["noise", "target"]
    small.rank("alpha beta", choices)
    large.rank("alpha beta", choices)

    t0 = time.perf_counter()
    for _ in range(80):
        small.rank("alpha beta", choices)
    small_ms = (time.perf_counter() - t0) * 1000.0 / 80

    t1 = time.perf_counter()
    for _ in range(80):
        large.rank("alpha beta", choices)
    large_ms = (time.perf_counter() - t1) * 1000.0 / 80

    t2 = time.perf_counter()
    for _ in range(80):
        large.rank("noise stream", ["target", "noise"])
    common_ms = (time.perf_counter() - t2) * 1000.0 / 80

    assert large_ms < small_ms * 8.0 + 2.0
    assert common_ms < large_ms * 8.0 + 2.0
    assert large.rank("alpha beta", choices)[0] == 1
    assert large.rank("noise stream", ["target", "noise"])[0] == 1


def test_micro_rank_model_loads_tiny_file_and_ranks_text(tmp_path: Path):
    mem = BrainMemory(window_size=4)
    mem.expose_text("python machine learning language data model", source="unit")
    mem.expose_text("css web page style layout color", source="unit")

    path = tmp_path / "rank_micro.pmr"
    mem.save_rank_micro_only(path)
    micro = MicroRankModel.load(path)

    pred, scores = micro.rank("machine learning language", ["CSS", "Python"])

    assert pred == 1
    assert scores[1] > scores[0]
    assert path.stat().st_size < 2048
    assert zlib.decompress(path.read_bytes()).startswith(b"PMR2")


def test_micro_rank_uses_one_shared_hash_space_for_file_modalities(tmp_path: Path):
    img = tmp_path / "sample.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (16).to_bytes(4, "big") + (8).to_bytes(4, "big") + b"payload")
    txt = tmp_path / "sample.txt"
    txt.write_text("alpha beta text memory", encoding="utf-8")

    mem = BrainMemory(window_size=4)
    mem.expose_file(img)
    mem.expose_file(txt)
    path = tmp_path / "rank_micro.pmr"
    mem.save_rank_micro_only(path)
    micro = MicroRankModel.load(path)

    pred, scores = micro.rank(str(img), [str(txt), str(img)])

    assert pred == 1
    assert scores[1] > scores[0]


def test_micro_rank_model_supports_32_bit_hash_export(tmp_path: Path):
    mem = BrainMemory(window_size=4)
    mem.expose_text("python machine learning language data model", source="unit")
    mem.expose_text("css web page style layout color", source="unit")

    path = tmp_path / "rank_micro_32.pmr"
    mem.save_rank_micro_only(path, hash_bits=32)
    micro = MicroRankModel.load(path)

    pred, scores = micro.rank("machine learning language", ["CSS", "Python"])

    assert micro.hash_bits == 32
    assert pred == 1
    assert scores[1] > scores[0]


def test_pmr2_varint_format_can_load_event_ids_above_65535():
    raw = bytearray(b"PMR2")
    write_varuint(raw, 16)
    write_varuint(raw, 1)
    write_varuint(raw, 123)
    write_varuint(raw, 1)
    write_varuint(raw, 70000)
    write_varuint(raw, 3)
    write_varuint(raw, 1)
    write_varuint(raw, 70000)
    write_varuint(raw, 1)
    write_varuint(raw, 123)
    micro = MicroRankModel.from_bytes(zlib.compress(bytes(raw), level=9))

    assert 70000 in micro.event_features
    assert micro.feature_rows[123] == [(70000, 3)]
