use pyo3::prelude::*;
use ahash::{AHashMap, AHashSet};
use std::collections::HashMap as StdHashMap;
use rayon::prelude::*;

fn data_scale(feature_count: usize, event_count: usize, row_count: usize, vocab_size: usize) -> usize {
    (feature_count + event_count + row_count + vocab_size).max(1)
}

fn dynamic_hdc_bits(feature_count: usize, event_count: usize, row_count: usize, vocab_size: usize) -> usize {
    let s = data_scale(feature_count, event_count, row_count, vocab_size) as f64;
    (((s + 1.0).log2().ceil() as usize) * (s.sqrt().ceil() as usize)).max(1)
}

fn dynamic_band_bits(bits: usize, row_count: usize, event_count: usize) -> usize {
    let s = (bits + row_count + event_count).max(1) as f64;
    (s.sqrt().ceil() as usize).max(1).min(bits.max(1))
}

fn dynamic_top_neighbors(row_count: usize, event_count: usize, vocab_size: usize) -> usize {
    let s = (row_count + event_count + vocab_size).max(1) as f64;
    (row_count.max(1) as f64).sqrt().ceil() as usize + (s + 1.0).log2().ceil() as usize
}

fn dynamic_hebb_rows(active_rows: usize, event_count: usize, row_count: usize) -> usize {
    let s = (active_rows + event_count + row_count).max(1) as f64;
    active_rows.max(1).min((s.sqrt().ceil() as usize) + ((s + 1.0).log2().ceil() as usize))
}

fn dynamic_pull_bits(bits: usize, amount: u32, row_weight: f64) -> usize {
    let strength = (amount.max(1) as f64) * row_weight.max(0.0);
    (((bits.max(1) as f64 + 1.0).log2() * (strength + 1.0).sqrt()).ceil() as usize).max(1).min(bits.max(1))
}

fn dynamic_similarity_floor(vals: &Vec<f64>) -> f64 {
    if vals.is_empty() { return 1.0; }
    let mut v = vals.clone();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len()/2]
}

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn tok_class(tok: &str) -> &'static str {
    match tok {
        "[NL]" => "[NL]",
        "[INDENT]" => "[INDENT]",
        "[DEDENT]" => "[DEDENT]",
        "[EOS]" => "[EOS]",
        "[BOS]" => "[BOS]",
        "[SEP]" => "[SEP]",
        "(" | ")" | "[" | "]" | "{" | "}" => "BRACKET",
        ":" | "," | ";" | "." => "PUNCT",
        "+" | "-" | "*" | "/" | "//" | "%" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "=" | "+=" | "-=" | "*=" | "/=" => "OP",
        _ => {
            if tok.chars().all(|c| c.is_ascii_digit()) { "NUM" }
            else if (tok.starts_with('"') && tok.ends_with('"')) || (tok.starts_with('\'') && tok.ends_with('\'')) { "STR" }
            else if tok.chars().next().map(|c| c.is_ascii_alphabetic() || c == '_').unwrap_or(false) { "WORD" }
            else { "SYM" }
        }
    }
}

fn words_for_bits(bits: usize) -> usize { bits.div_ceil(64) }
fn last_mask(bits: usize) -> u64 {
    let rem = bits % 64;
    if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 }
}
fn trim(mut hv: Vec<u64>, bits: usize) -> Vec<u64> {
    let words = words_for_bits(bits);
    hv.resize(words, 0);
    if words > 0 { hv[words - 1] &= last_mask(bits); }
    hv
}
fn base_hv(f: &str, bits: usize) -> Vec<u64> {
    let seed = fnv1a64(f);
    let words = words_for_bits(bits);
    let mut out = Vec::with_capacity(words);
    for word in 0..words {
        out.push(splitmix64(seed.wrapping_add((word as u64).wrapping_mul(0xD1B54A32D192ED03))));
    }
    trim(out, bits)
}
fn expand_hv(f: &str, hv: &Vec<u64>, bits: usize) -> Vec<u64> {
    let words = words_for_bits(bits);
    if hv.len() >= words { return trim(hv.clone(), bits); }
    let base = base_hv(f, bits);
    let mut out = hv.clone();
    out.resize(words, 0);
    for i in hv.len()..words { out[i] = base[i]; }
    trim(out, bits)
}
fn get_bit(hv: &Vec<u64>, pos: usize) -> bool {
    let w = pos / 64;
    let b = pos % 64;
    w < hv.len() && ((hv[w] >> b) & 1) == 1
}
fn set_bit_to(dst: &mut Vec<u64>, pos: usize, value: bool) {
    let w = pos / 64;
    let b = pos % 64;
    if w >= dst.len() { dst.resize(w + 1, 0); }
    if value { dst[w] |= 1u64 << b; } else { dst[w] &= !(1u64 << b); }
}
fn xor_popcount(a: &Vec<u64>, b: &Vec<u64>, bits: usize) -> u32 {
    let words = words_for_bits(bits);
    let mut dist = 0u32;
    for i in 0..words {
        let aw = *a.get(i).unwrap_or(&0);
        let bw = *b.get(i).unwrap_or(&0);
        let mut x = aw ^ bw;
        if i + 1 == words { x &= last_mask(bits); }
        dist += x.count_ones();
    }
    dist
}
fn parse_decimal_to_words(s: &str, bits: usize) -> Vec<u64> {
    // Decimal compatibility from Python pickle. Slow but used only on restore.
    let mut words = vec![0u64; words_for_bits(bits)];
    for ch in s.bytes() {
        if !(b'0'..=b'9').contains(&ch) { continue; }
        let digit = (ch - b'0') as u128;
        let mut carry = digit;
        for w in words.iter_mut() {
            let val = (*w as u128) * 10u128 + carry;
            *w = val as u64;
            carry = val >> 64;
        }
    }
    trim(words, bits)
}
fn bucket_keys(bits: usize, hv: &Vec<u64>) -> Vec<u64> {
    let band_bits = dynamic_band_bits(bits, 0, 0);
    let bands = bits.div_ceil(band_bits);
    let mut out = Vec::with_capacity(bands);
    for band in 0..bands {
        let mut seg: u64 = 0;
        for j in 0..band_bits {
            let pos = band * band_bits + j;
            if pos >= bits { break; }
            if get_bit(hv, pos) { seg |= 1u64 << j; }
        }
        out.push(((band as u64) << 32) | seg);
    }
    out
}
fn rows_from_plain(features: Vec<String>) -> Vec<(String, f64)> {
    features.into_iter().map(|f| (f, 1.0)).collect()
}
fn hv_key(bits: usize, hv: &Vec<u64>) -> String {
    let mut s = bits.to_string();
    s.push(':');
    for w in hv.iter().rev() { s.push_str(&format!("{:016x}", w)); }
    s
}

#[pyclass]
pub struct RustHebbianHdcCountEvidence {
    rows: AHashMap<String, AHashMap<String, u32>>,
    totals: AHashMap<String, u32>,
    hdc_rows: AHashMap<String, AHashMap<String, u32>>,
    hdc_totals: AHashMap<String, u32>,
    hdc_vecs: AHashMap<String, (usize, Vec<u64>)>,
    buckets: AHashMap<u64, Vec<String>>,
    feature_hv: AHashMap<String, Vec<u64>>,
    hdc_bits: usize,
    updates: u64,
    resize_count: u64,
}

impl RustHebbianHdcCountEvidence {
    fn target_bits(&self) -> usize {
        dynamic_hdc_bits(self.rows.len() + self.feature_hv.len(), self.updates as usize, self.hdc_rows.len(), self.vocab_size())
    }
    fn vocab_size(&self) -> usize {
        let mut set = AHashSet::new();
        for row in self.rows.values() { for k in row.keys() { set.insert(k.clone()); } }
        set.len()
    }
    fn maybe_resize(&mut self) {
        let target = self.target_bits();
        if target > self.hdc_bits {
            self.hdc_bits = target;
            self.resize_count += 1;
            self.rebuild_buckets();
        }
    }
    fn rebuild_buckets(&mut self) {
        self.buckets.clear();
        for (k, (bits, hv)) in self.hdc_vecs.iter() {
            for bk in bucket_keys(*bits, hv) {
                let bucket = self.buckets.entry(bk).or_insert_with(Vec::new);
                if !bucket.contains(k) { bucket.push(k.clone()); }
            }
        }
    }
    fn get_hv(&self, f: &str, bits: usize) -> Vec<u64> {
        self.feature_hv.get(f).map(|v| expand_hv(f, v, bits)).unwrap_or_else(|| base_hv(f, bits))
    }
    fn ensure_hv(&mut self, f: &str) -> Vec<u64> {
        let hv = self.feature_hv.get(f).map(|v| expand_hv(f, v, self.hdc_bits)).unwrap_or_else(|| base_hv(f, self.hdc_bits));
        self.feature_hv.insert(f.to_string(), hv.clone());
        hv
    }
    fn hdc_vector(&self, features: &Vec<(String, f64)>, bits: usize) -> Vec<u64> {
        if features.is_empty() { return vec![0; words_for_bits(bits)]; }
        let mut acc = vec![0f64; bits];
        for (f, w) in features.iter() {
            if *w <= 0.0 { continue; }
            let hv = self.get_hv(f, bits);
            for b in 0..bits { if get_bit(&hv, b) { acc[b] += *w; } else { acc[b] -= *w; } }
        }
        let mut hv = vec![0u64; words_for_bits(bits)];
        for i in 0..bits { if acc[i] >= 0.0 { set_bit_to(&mut hv, i, true); } }
        trim(hv, bits)
    }
    fn pull_bits(&self, src: &Vec<u64>, dst: &Vec<u64>, seed: u64, bits_to_pull: usize) -> Vec<u64> {
        let bits = self.hdc_bits;
        let mut out = expand_hv("", src, bits);
        let d = trim(dst.clone(), bits);
        if bits_to_pull == 0 { return out; }
        let mut diff_positions: Vec<usize> = Vec::new();
        let mut x = seed;
        let mut tries = 0usize;
        while diff_positions.len() < bits_to_pull && tries < bits * 4 {
            x = splitmix64(x.wrapping_add(tries as u64).wrapping_add(0x9E3779B97F4A7C15));
            let pos = (x as usize) % bits;
            if get_bit(&out, pos) != get_bit(&d, pos) && !diff_positions.contains(&pos) { diff_positions.push(pos); }
            tries += 1;
        }
        if diff_positions.is_empty() {
            for pos in 0..bits { if get_bit(&out, pos) != get_bit(&d, pos) { diff_positions.push(pos); break; } }
        }
        for pos in diff_positions { set_bit_to(&mut out, pos, get_bit(&d, pos)); }
        trim(out, bits)
    }
    fn hebbian_update(&mut self, features: &Vec<(String, f64)>, token: &str, amount: u32) {
        if features.is_empty() { return; }
        let mut selected = features.clone();
        selected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep = dynamic_hebb_rows(selected.len(), self.updates as usize, self.rows.len() + self.hdc_rows.len());
        if selected.len() > keep { selected.truncate(keep); }
        let ctx_hv = self.hdc_vector(&selected, self.hdc_bits);
        let y_key = format!("Y|{}", token);
        let yc_key = format!("YC|{}", tok_class(token));
        let mut y_hv = self.ensure_hv(&y_key);
        let yc_hv = self.ensure_hv(&yc_key);
        let r = 0u32;
        let seed_y = fnv1a64(&format!("{}|CTX|{}|{}", y_key, self.updates, r));
        y_hv = self.pull_bits(&y_hv, &ctx_hv, seed_y, dynamic_pull_bits(self.hdc_bits, amount, 1.0));
        y_hv = self.pull_bits(&y_hv, &yc_hv, seed_y ^ 0xA5A5A5A5A5A5A5A5u64, dynamic_pull_bits(self.hdc_bits, amount, 0.5));
        self.feature_hv.insert(y_key.clone(), y_hv.clone());
        for (f, w) in selected.iter() {
            if *w <= 0.04 { continue; }
            let mut fhv = self.ensure_hv(f);
            let bits = dynamic_pull_bits(self.hdc_bits, amount, *w);
            let seed_f = fnv1a64(&format!("{}|{}|{}|{}", f, y_key, self.updates, r));
            fhv = self.pull_bits(&fhv, &y_hv, seed_f, bits);
            if *w >= 0.35 { fhv = self.pull_bits(&fhv, &yc_hv, seed_f ^ 0xD1B54A32D192ED03u64, dynamic_pull_bits(self.hdc_bits, amount, 0.5)); }
            self.feature_hv.insert(f.clone(), fhv);
        }
    }
}

#[pymethods]
impl RustHebbianHdcCountEvidence {
    #[new]
    fn new() -> Self {
        Self {
            rows: AHashMap::new(), totals: AHashMap::new(), hdc_rows: AHashMap::new(), hdc_totals: AHashMap::new(),
            hdc_vecs: AHashMap::new(), buckets: AHashMap::new(), feature_hv: AHashMap::new(),
            hdc_bits: dynamic_hdc_bits(0, 0, 0, 0), updates: 0, resize_count: 0,
        }
    }

    fn configure_dynamic(&mut self, feature_count: usize, event_count: usize, row_count: usize, vocab_size: usize) {
        let target = dynamic_hdc_bits(feature_count + self.feature_hv.len(), event_count.max(self.updates as usize), row_count.max(self.hdc_rows.len()), vocab_size.max(self.vocab_size()));
        if target > self.hdc_bits { self.hdc_bits = target; self.resize_count += 1; self.rebuild_buckets(); }
    }

    fn set_feature_hv_decimal(&mut self, items: Vec<(String, String)>) {
        for (k, v) in items { self.feature_hv.insert(k, parse_decimal_to_words(&v, self.hdc_bits)); }
    }

    fn update_features(&mut self, features: Vec<String>, token: String) { self.update_features_weighted(rows_from_plain(features), token, 1); }

    fn update_features_weighted(&mut self, features: Vec<(String, f64)>, token: String, amount: u32) {
        let amt = amount.max(1);
        if features.is_empty() { return; }
        self.maybe_resize();
        self.hebbian_update(&features, &token, amt);
        for (f, _w) in features.iter() {
            let row = self.rows.entry(f.clone()).or_insert_with(AHashMap::new);
            *row.entry(token.clone()).or_insert(0) += amt;
            *self.totals.entry(f.clone()).or_insert(0) += amt;
        }
        let hv = self.hdc_vector(&features, self.hdc_bits);
        if hv.iter().any(|x| *x != 0) {
            let key = hv_key(self.hdc_bits, &hv);
            let row = self.hdc_rows.entry(key.clone()).or_insert_with(AHashMap::new);
            *row.entry(token.clone()).or_insert(0) += amt;
            *self.hdc_totals.entry(key.clone()).or_insert(0) += amt;
            self.hdc_vecs.entry(key.clone()).or_insert((self.hdc_bits, hv.clone()));
            for bk in bucket_keys(self.hdc_bits, &hv) {
                let bucket = self.buckets.entry(bk).or_insert_with(Vec::new);
                if !bucket.contains(&key) { bucket.push(key.clone()); }
            }
        }
        self.updates += amt as u64;
        self.maybe_resize();
    }

    fn update_many(&mut self, batch: Vec<(Vec<String>, String)>) {
        for (features, token) in batch { self.update_features_weighted(rows_from_plain(features), token, 1); }
    }

    fn update_many_weighted(&mut self, batch: Vec<(Vec<(String, f64)>, String, u32)>) {
        for (features, token, amount) in batch { self.update_features_weighted(features, token, amount); }
    }

    fn score_features(&self, features: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> {
        if features.is_empty() { return Vec::new(); }
        let exact = features.par_iter()
            .fold(|| AHashMap::<String, f64>::new(), |mut acc, (f, w)| {
                if let Some(row) = self.rows.get(f) {
                    let total = (*self.totals.get(f).unwrap_or(&1)).max(1) as f64;
                    for (tok, cnt) in row.iter() { *acc.entry(tok.clone()).or_insert(0.0) += *w * (*cnt as f64 / total); }
                }
                acc
            })
            .reduce(|| AHashMap::<String, f64>::new(), |mut a, b| { for (tok, s) in b { *a.entry(tok).or_insert(0.0) += s; } a });

        let q = self.hdc_vector(&features, self.hdc_bits);
        let mut cand: AHashSet<String> = AHashSet::new();
        if q.iter().any(|x| *x != 0) {
            for bk in bucket_keys(self.hdc_bits, &q) { if let Some(v) = self.buckets.get(&bk) { for k in v { cand.insert(k.clone()); } } }
            if cand.len() < dynamic_top_neighbors(self.hdc_rows.len(), self.updates as usize, self.vocab_size()) && self.hdc_rows.len() <= dynamic_hebb_rows(self.hdc_rows.len(), self.updates as usize, self.rows.len()) { for k in self.hdc_rows.keys() { cand.insert(k.clone()); } }
        }
        let mut sims: Vec<(f64, String)> = cand.into_iter().filter_map(|key| {
            let (bits, hv) = self.hdc_vecs.get(&key)?;
            let common_bits = (*bits).min(self.hdc_bits).max(1);
            let dist = xor_popcount(&q, hv, common_bits) as f64;
            let sim = 1.0 - dist / (common_bits as f64);
            Some((sim, key))
        }).collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let topn = dynamic_top_neighbors(self.hdc_rows.len(), self.updates as usize, self.vocab_size());
        if sims.len() > topn { sims.truncate(topn); }

        let vals: Vec<f64> = sims.iter().map(|x| x.0).collect();
        let floor = dynamic_similarity_floor(&vals);
        let top = vals.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let span = (top - floor).max(1e-9);
        let hdc = sims.par_iter()
            .fold(|| AHashMap::<String, f64>::new(), |mut acc, (sim, key)| {
                if *sim >= floor {
                    if let Some(row) = self.hdc_rows.get(key) {
                        let total = (*self.hdc_totals.get(key).unwrap_or(&1)).max(1) as f64;
                        let w = ((*sim - floor) / span).max(0.0);
                        if w > 0.0 { for (tok, cnt) in row.iter() { *acc.entry(tok.clone()).or_insert(0.0) += w * (*cnt as f64 / total); } }
                    }
                }
                acc
            })
            .reduce(|| AHashMap::<String, f64>::new(), |mut a, b| { for (tok, s) in b { *a.entry(tok).or_insert(0.0) += s; } a });

        let mut merged = exact;
        for (tok, s) in hdc { *merged.entry(tok).or_insert(0.0) += s; }
        let mut v: Vec<(String, f64)> = merged.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        v.truncate(top_k);
        v
    }

    fn get_statistics(&self) -> StdHashMap<String, u64> {
        let mut m = StdHashMap::new();
        m.insert("feature_count".to_string(), self.rows.len() as u64);
        m.insert("feature_next_entries".to_string(), self.rows.values().map(|r| r.len() as u64).sum());
        m.insert("hdc_row_count".to_string(), self.hdc_rows.len() as u64);
        m.insert("hdc_next_entries".to_string(), self.hdc_rows.values().map(|r| r.len() as u64).sum());
        m.insert("hebbian_feature_vectors".to_string(), self.feature_hv.len() as u64);
        m.insert("updates".to_string(), self.updates);
        m.insert("hdc_bits".to_string(), self.hdc_bits as u64);
        m.insert("hdc_resize_count".to_string(), self.resize_count);
        m
    }

    fn clear(&mut self) {
        self.rows.clear(); self.totals.clear(); self.hdc_rows.clear(); self.hdc_totals.clear(); self.hdc_vecs.clear(); self.buckets.clear(); self.feature_hv.clear();
        self.updates = 0; self.resize_count = 0; self.hdc_bits = dynamic_hdc_bits(0, 0, 0, 0);
    }
}

#[pyclass]
pub struct RustHdcCountEvidence { inner: RustHebbianHdcCountEvidence }
#[pymethods]
impl RustHdcCountEvidence {
    #[new]
    fn new() -> Self { Self { inner: RustHebbianHdcCountEvidence::new() } }
    fn configure_dynamic(&mut self, feature_count: usize, event_count: usize, row_count: usize, vocab_size: usize) { self.inner.configure_dynamic(feature_count, event_count, row_count, vocab_size); }
    fn set_feature_hv_decimal(&mut self, items: Vec<(String, String)>) { self.inner.set_feature_hv_decimal(items); }
    fn update_features(&mut self, features: Vec<String>, token: String) { self.inner.update_features(features, token); }
    fn update_features_weighted(&mut self, features: Vec<(String, f64)>, token: String, amount: u32) { self.inner.update_features_weighted(features, token, amount); }
    fn update_many(&mut self, batch: Vec<(Vec<String>, String)>) { self.inner.update_many(batch); }
    fn update_many_weighted(&mut self, batch: Vec<(Vec<(String, f64)>, String, u32)>) { self.inner.update_many_weighted(batch); }
    fn score_features(&self, features: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> { self.inner.score_features(features, top_k) }
    fn clear(&mut self) { self.inner.clear(); }
    fn get_statistics(&self) -> StdHashMap<String, u64> { self.inner.get_statistics() }
}

#[pyclass]
pub struct RustCountEvidence { inner: RustHebbianHdcCountEvidence }
#[pymethods]
impl RustCountEvidence {
    #[new]
    fn new() -> Self { Self { inner: RustHebbianHdcCountEvidence::new() } }
    fn configure_dynamic(&mut self, feature_count: usize, event_count: usize, row_count: usize, vocab_size: usize) { self.inner.configure_dynamic(feature_count, event_count, row_count, vocab_size); }
    fn set_feature_hv_decimal(&mut self, items: Vec<(String, String)>) { self.inner.set_feature_hv_decimal(items); }
    fn update_features(&mut self, features: Vec<String>, token: String) { self.inner.update_features(features, token); }
    fn update_features_weighted(&mut self, features: Vec<(String, f64)>, token: String, amount: u32) { self.inner.update_features_weighted(features, token, amount); }
    fn update_many(&mut self, batch: Vec<(Vec<String>, String)>) { self.inner.update_many(batch); }
    fn update_many_weighted(&mut self, batch: Vec<(Vec<(String, f64)>, String, u32)>) { self.inner.update_many_weighted(batch); }
    fn score_features(&self, features: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> { self.inner.score_features(features, top_k) }
    fn clear(&mut self) { self.inner.clear(); }
    fn get_statistics(&self) -> StdHashMap<String, u64> { self.inner.get_statistics() }
}

#[pyclass]
pub struct RustLogitTables { inner: RustHebbianHdcCountEvidence }
#[pymethods]
impl RustLogitTables {
    #[new]
    fn new() -> Self { Self { inner: RustHebbianHdcCountEvidence::new() } }
    fn configure_dynamic(&mut self, feature_count: usize, event_count: usize, row_count: usize, vocab_size: usize) { self.inner.configure_dynamic(feature_count, event_count, row_count, vocab_size); }
    fn set_feature_hv_decimal(&mut self, items: Vec<(String, String)>) { self.inner.set_feature_hv_decimal(items); }
    fn update_features(&mut self, features: Vec<String>, token: String) { self.inner.update_features(features, token); }
    fn update_features_weighted(&mut self, features: Vec<(String, f64)>, token: String, amount: u32) { self.inner.update_features_weighted(features, token, amount); }
    fn update_many(&mut self, batch: Vec<(Vec<String>, String)>) { self.inner.update_many(batch); }
    fn update_many_weighted(&mut self, batch: Vec<(Vec<(String, f64)>, String, u32)>) { self.inner.update_many_weighted(batch); }
    fn score_features(&self, features: Vec<(String, f64)>, top_k: usize) -> Vec<(String, f64)> { self.inner.score_features(features, top_k) }
    fn clear(&mut self) { self.inner.clear(); }
    fn get_statistics(&self) -> StdHashMap<String, u64> { self.inner.get_statistics() }
}


#[pyclass]
pub struct RustFrozenExactScorer {
    tokens: Vec<String>,
    row_index: AHashMap<u64, usize>,
    row_ptr: Vec<usize>,
    token_ids: Vec<usize>,
    values: Vec<f32>,
}

#[pymethods]
impl RustFrozenExactScorer {
    #[new]
    fn new(tokens: Vec<String>, row_ids: Vec<u64>, row_ptr: Vec<usize>, token_ids: Vec<usize>, values: Vec<f32>) -> Self {
        let mut row_index = AHashMap::new();
        for (i, rid) in row_ids.iter().enumerate() { row_index.insert(*rid, i); }
        Self { tokens, row_index, row_ptr, token_ids, values }
    }

    fn score_row_ids(&self, row_ids: Vec<u64>, weights: Vec<f64>, top_k: usize) -> Vec<(String, f64)> {
        // P87: single-prompt scoring uses a dense/touched accumulator instead of
        // per-row Rayon hash-map reduction.  Rayon overhead dominates for the
        // small active-row sets typical of next-token decoding.  Batch-level
        // parallelism is handled by RustFrozenNextEngine::batch_predict_text.
        self.score_ids_dense_internal(&row_ids, &weights, top_k)
    }

    fn score_row_ids_parallel(&self, row_ids: Vec<u64>, weights: Vec<f64>, top_k: usize) -> Vec<(String, f64)> {
        // Explicit old-style parallel scorer kept for very large one-shot calls.
        self.score_ids_parallel_internal(&row_ids, &weights, top_k)
    }

    fn score_top1_row_ids(&self, row_ids: Vec<u64>, weights: Vec<f64>) -> (String, f64) {
        self.score_top1_internal(&row_ids, &weights).unwrap_or_else(|| (String::new(), 0.0))
    }

    fn get_statistics(&self) -> StdHashMap<String, u64> {
        let mut m = StdHashMap::new();
        m.insert("tokens".to_string(), self.tokens.len() as u64);
        m.insert("rows".to_string(), self.row_index.len() as u64);
        m.insert("entries".to_string(), self.token_ids.len() as u64);
        m.insert("values_bytes".to_string(), (self.values.len() * std::mem::size_of::<f32>()) as u64);
        m
    }
}


impl RustFrozenExactScorer {
    fn accumulate_dense(&self, row_ids: &[u64], weights: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let mut scores = vec![0.0f64; self.tokens.len()];
        let mut touched = Vec::<usize>::new();
        for (rid, w) in row_ids.iter().zip(weights.iter()) {
            if *w <= 0.0 { continue; }
            if let Some(idx) = self.row_index.get(rid) {
                if *idx + 1 < self.row_ptr.len() {
                    let start = self.row_ptr[*idx];
                    let end = self.row_ptr[*idx + 1].min(self.token_ids.len()).min(self.values.len());
                    for j in start..end {
                        let tid = self.token_ids[j];
                        if tid >= scores.len() { continue; }
                        if scores[tid] == 0.0 { touched.push(tid); }
                        scores[tid] += *w * (self.values[j] as f64);
                    }
                }
            }
        }
        (scores, touched)
    }

    fn score_top1_internal(&self, row_ids: &[u64], weights: &[f64]) -> Option<(String, f64)> {
        let (scores, touched) = self.accumulate_dense(row_ids, weights);
        let mut best_tid: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for tid in touched {
            let s = scores[tid];
            if s == 0.0 { continue; }
            if best_tid.is_none() || s > best_score || (s == best_score && self.tokens[tid] < self.tokens[best_tid.unwrap()]) {
                best_tid = Some(tid);
                best_score = s;
            }
        }
        best_tid.and_then(|tid| self.tokens.get(tid).map(|tok| (tok.clone(), best_score)))
    }

    fn score_ids_dense_internal(&self, row_ids: &[u64], weights: &[f64], top_k: usize) -> Vec<(String, f64)> {
        if top_k == 1 {
            return self.score_top1_internal(row_ids, weights).into_iter().collect();
        }
        let (scores, touched) = self.accumulate_dense(row_ids, weights);
        let mut out: Vec<(String, f64)> = touched.into_iter()
            .filter_map(|tid| {
                let s = scores[tid];
                if s == 0.0 { None } else { self.tokens.get(tid).map(|tok| (tok.clone(), s)) }
            })
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        if top_k > 0 && out.len() > top_k { out.truncate(top_k); }
        out
    }

    fn score_ids_parallel_internal(&self, row_ids: &[u64], weights: &[f64], top_k: usize) -> Vec<(String, f64)> {
        let merged = row_ids.par_iter().zip(weights.par_iter())
            .fold(|| AHashMap::<usize, f64>::new(), |mut acc, (rid, w)| {
                if *w > 0.0 {
                    if let Some(idx) = self.row_index.get(rid) {
                        if *idx + 1 < self.row_ptr.len() {
                            let start = self.row_ptr[*idx];
                            let end = self.row_ptr[*idx + 1].min(self.token_ids.len()).min(self.values.len());
                            for j in start..end {
                                let tid = self.token_ids[j];
                                *acc.entry(tid).or_insert(0.0) += *w * (self.values[j] as f64);
                            }
                        }
                    }
                }
                acc
            })
            .reduce(|| AHashMap::<usize, f64>::new(), |mut a, b| { for (tid, s) in b { *a.entry(tid).or_insert(0.0) += s; } a });
        let mut out: Vec<(String, f64)> = merged.into_iter()
            .filter_map(|(tid, s)| self.tokens.get(tid).map(|tok| (tok.clone(), s)))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        if top_k > 0 && out.len() > top_k { out.truncate(top_k); }
        out
    }

    fn score_ids_internal(&self, row_ids: &[u64], weights: &[f64], top_k: usize) -> Vec<(String, f64)> {
        self.score_ids_dense_internal(row_ids, weights, top_k)
    }
}

fn is_word_token(tok: &str) -> bool {
    let mut chars = tok.chars();
    match chars.next() { Some(c) if c.is_ascii_alphabetic() || c == '_' => {}, _ => return false }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
fn has_alnum(tok: &str) -> bool { tok.chars().any(|c| c.is_ascii_alphanumeric() || c == '_') }
fn is_stop(tok: &str) -> bool {
    matches!(tok,
        "the"|"a"|"an"|"and"|"or"|"to"|"of"|"in"|"on"|"for"|"with"|"by"|"is"|"are"|"be"|"as"|"that"|"this"|
        "write"|"python"|"function"|"given"|"return"|"returns"|"which"|"takes"|"input"|"output"|"from"|"using"|
        "create"|"make"|"find"|"get"|"check"|"whether"|"true"|"false"|"list"|"array"|"string"|"number"|"numbers"|"integer"|"integers"|
        "two"|"three"|"four"|"five"|"value"|"values"|"item"|"items"|"element"|"elements"|"result"|"res")
}
fn tokenize_fast(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.replace("\r\n", "\n").replace('\r', "\n").chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' { out.push("[NL]".to_string()); i += 1; continue; }
        if c.is_whitespace() { i += 1; continue; }
        if c.is_ascii_alphabetic() || c == '_' {
            let start=i; i+=1; while i<chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i]=='_') { i+=1; }
            out.push(chars[start..i].iter().collect()); continue;
        }
        if c.is_ascii_digit() {
            let start=i; i+=1; while i<chars.len() && (chars[i].is_ascii_digit() || chars[i]=='.') { i+=1; }
            out.push(chars[start..i].iter().collect()); continue;
        }
        if c == '"' || c == '\'' {
            let quote=c; let start=i; i+=1; while i<chars.len() { let cc=chars[i]; i+=1; if cc==quote { break; } }
            out.push(chars[start..i.min(chars.len())].iter().collect()); continue;
        }
        if i+1<chars.len() {
            let two: String = chars[i..i+2].iter().collect();
            if matches!(two.as_str(), "=="|"!="|"<="|">="|"//"|"+="|"-="|"*="|"/="|"<<"|">>"|"->") {
                out.push(two); i += 2; continue;
            }
        }
        out.push(c.to_string()); i += 1;
    }
    out
}
fn copy_tokens(input_tokens: &[String]) -> Vec<String> {
    let mut primary = Vec::<String>::new();
    let mut secondary = Vec::<String>::new();
    let mut seen = AHashSet::<String>::new();
    let mut depth: i32 = 0;
    for t in input_tokens {
        match t.as_str() { "("|"["|"{" => { depth += 1; continue; }, ")"|"]"|"}" => { depth = (depth-1).max(0); continue; }, _ => {} }
        if matches!(t.as_str(), "[NL]"|"[INDENT]"|"[DEDENT]"|"[BOS]"|"[SEP]"|"[EOS]") { continue; }
        let low = t.to_lowercase();
        let ok = is_word_token(t) || t.chars().all(|c| c.is_ascii_digit()) || t.starts_with('"') || t.starts_with('\'');
        if !ok { continue; }
        let key = if is_word_token(t) { low.clone() } else { t.clone() };
        if seen.contains(&key) { continue; }
        if !is_stop(&low) && low.len() >= 2 { primary.push(t.clone()); seen.insert(key); }
        else if depth > 0 && is_word_token(t) { secondary.push(t.clone()); seen.insert(key); }
        else if !is_stop(&low) && low.len() == 1 { secondary.push(t.clone()); seen.insert(key); }
    }
    primary.extend(secondary); primary
}
fn p89_data_feature_budget(available: usize, prompt_tokens: usize, active_rows: usize, se: usize, sf: usize, sr: usize, sv: usize) -> usize {
    if available == 0 { return 0; }
    let s = (available + prompt_tokens + active_rows + se + sf + sr + sv).max(1) as f64;
    available.min((s.sqrt().ceil() as usize) + ((s + 1.0).log2().ceil() as usize)).max(1)
}
fn p89_take<T: Clone>(items: &[T], prompt_tokens: usize, active_rows: usize, se: usize, sf: usize, sr: usize, sv: usize) -> Vec<T> {
    let n = p89_data_feature_budget(items.len(), prompt_tokens, active_rows, se, sf, sr, sv).min(items.len());
    items.iter().take(n).cloned().collect()
}

fn prompt_features_scaled(input_tokens: &[String], se: usize, sf: usize, sr: usize, sv: usize) -> Vec<String> {
    let mut feats=Vec::<String>::new(); let mut seen=AHashSet::<String>::new();
    let lows: Vec<String> = input_tokens.iter().filter(|t| !matches!(t.as_str(), "[NL]"|"[INDENT]"|"[DEDENT]")).map(|t| t.to_lowercase()).collect();
    let classes: Vec<String> = input_tokens.iter().filter(|t| !matches!(t.as_str(), "[NL]"|"[INDENT]"|"[DEDENT]")).map(|t| tok_class(t).to_string()).collect();
    let mut rare=Vec::<String>::new(); let mut seen_rare=AHashSet::<String>::new();
    for t in &lows { if t.len()>=3 && !is_stop(t) && has_alnum(t) && !seen_rare.contains(t) { seen_rare.insert(t.clone()); rare.push(t.clone()); } }
    let mut rare_sorted=rare.clone(); rare_sorted.sort();
    let mut onechars=Vec::<String>::new();
    for t in &lows { if t.len()==1 && has_alnum(t) && !onechars.contains(t) { onechars.push(t.clone()); } }
    if rare.is_empty() { for t in &lows { if t.len()==1 && has_alnum(t) && !seen_rare.contains(t) { seen_rare.insert(t.clone()); rare.push(t.clone()); } } rare_sorted=rare.clone(); rare_sorted.sort(); }
    if !rare.is_empty() || !onechars.is_empty() {
        let rare_budget = p89_data_feature_budget(rare.len(), lows.len(), feats.len(), se, sf, sr, sv);
        let one_budget = p89_data_feature_budget(onechars.len(), lows.len(), feats.len(), se, sf, sr, sv);
        let mut sig_base: Vec<String> = rare.iter().take(rare_budget).cloned().collect();
        for c in onechars.iter().take(one_budget) { sig_base.push(format!("ch:{}", c)); }
        let rs_budget = p89_data_feature_budget(rare_sorted.len(), lows.len(), feats.len(), se, sf, sr, sv);
        let sorted_char_budget = p89_data_feature_budget(onechars.len(), lows.len(), feats.len(), se, sf, sr, sv);
        let mut bag_base: Vec<String> = rare_sorted.iter().take(rs_budget).cloned().collect();
        let mut sorted_chars=onechars.clone(); sorted_chars.sort();
        for c in sorted_chars.iter().take(sorted_char_budget) { bag_base.push(format!("ch:{}",c)); }
        feats.push(format!("sig:first:{}", sig_base.join("|")));
        feats.push(format!("sig:bag:{}", bag_base.join("|")));
        for c in p89_take(&onechars, lows.len(), feats.len(), se, sf, sr, sv) { feats.push(format!("char:{}",c)); }
        for (i,x) in p89_take(&rare, lows.len(), feats.len(), se, sf, sr, sv).iter().enumerate() { feats.push(format!("rare{}:{}",i,x)); feats.push(format!("rare:{}",x)); if x.len()>=4 { feats.push(format!("rpre4:{}", &x[0..4.min(x.len())])); let start=x.len().saturating_sub(4); feats.push(format!("rsuf4:{}", &x[start..])); } }
        let n_r2 = rare.len().saturating_sub(1).min(p89_data_feature_budget(rare.len(), lows.len(), feats.len(), se, sf, sr, sv));
        for i in 0..n_r2 { feats.push(format!("rare2:{}|{}", rare[i], rare[i+1])); }
        let n_bag = rare_sorted.len().min(p89_data_feature_budget(rare_sorted.len(), lows.len(), feats.len(), se, sf, sr, sv));
        for i in 0..n_bag { for j in (i+1)..n_bag { feats.push(format!("bag2:{}|{}", rare_sorted[i], rare_sorted[j])); } }
    } else { feats.push("prompt:no_rare".to_string()); }
    let n_lows = lows.len().min(p89_data_feature_budget(lows.len(), lows.len(), feats.len(), se, sf, sr, sv));
    for (i,t) in lows.iter().take(n_lows).enumerate() {
        if t.len()>=3 && !is_stop(t) && has_alnum(t) { feats.push(format!("pt:{}",t)); feats.push(format!("p3:{}",&t[0..3.min(t.len())])); let start=t.len().saturating_sub(3); feats.push(format!("p3e:{}",&t[start..])); }
        if i < classes.len() { feats.push(format!("pc:{}",classes[i])); }
    }
    let n_pt2 = lows.len().saturating_sub(1).min(p89_data_feature_budget(lows.len(), lows.len(), feats.len(), se, sf, sr, sv));
    for i in 0..n_pt2 { let a=&lows[i]; let b=&lows[i+1]; if (!is_stop(a)||!is_stop(b)) && a.len()>=2 && b.len()>=2 { feats.push(format!("pt2:{}|{}",a,b)); } }
    let n_pt3 = lows.len().saturating_sub(2).min(p89_data_feature_budget(lows.len(), lows.len(), feats.len(), se, sf, sr, sv));
    for i in 0..n_pt3 { let a=&lows[i]; let b=&lows[i+1]; let c=&lows[i+2]; if [a,b,c].iter().any(|x| !is_stop(x) && x.len()>=3) { feats.push(format!("pt3:{}|{}|{}",a,b,c)); } }
    let mut out=Vec::new();
    for f in feats { if !f.is_empty() && !seen.contains(&f) { seen.insert(f.clone()); out.push(f); } }
    out
}
fn template_id(feature: &str) -> String {
    let head = feature.split('|').next().unwrap_or("");
    if head.starts_with('R') && head.len()>1 && head[1..].chars().all(|c| c.is_ascii_digit()) { return "R*".to_string(); }
    if head.starts_with("RC") && head.len()>2 && head[2..].chars().all(|c| c.is_ascii_digit()) { return "RC*".to_string(); }
    head.to_string()
}
fn detok(tokens: &[String]) -> String {
    let mut out=String::new(); let mut line_start=true;
    for tok in tokens {
        if tok == "[NL]" { out.push('\n'); line_start=true; continue; }
        if tok == "[INDENT]" { if line_start { out.push_str("    "); } else { out.push(' '); } continue; }
        if tok == "[DEDENT]" { continue; }
        if out.is_empty() || line_start { out.push_str(tok); line_start=false; continue; }
        if matches!(tok.as_str(), ")"|"]"|"}"|","|":"|";"|".") { out.push_str(tok); }
        else if out.ends_with('(')||out.ends_with('[')||out.ends_with('{')||out.ends_with('.') { out.push_str(tok); }
        else if matches!(tok.as_str(), "("|"["|"{") { out.push_str(tok); }
        else if matches!(tok.as_str(), "+"|"-"|"*"|"/"|"//"|"%"|"="|"=="|"!="|"<"|">"|"<="|">="|"and"|"or") { out.push(' '); out.push_str(tok); out.push(' '); }
        else { if !out.ends_with(' ') { out.push(' '); } out.push_str(tok); }
    }
    out
}

#[pyclass]
pub struct RustFrozenNextEngine {
    scorer: RustFrozenExactScorer,
    gate_map: AHashMap<String, f64>,
    // P89: data-scale metadata copied from the Python training object at freeze
    // time.  This is not hardware capacity; it is the learned data state used
    // by Python _data_feature_budget/_dynamic_context_window.
    scale_events: usize,
    scale_features: usize,
    scale_rows: usize,
    scale_vocab: usize,
}

#[pymethods]
impl RustFrozenNextEngine {
    #[new]
    fn new(tokens: Vec<String>, row_ids: Vec<u64>, row_ptr: Vec<usize>, token_ids: Vec<usize>, values: Vec<f32>, gate_items: Vec<(String, f64)>) -> Self {
        let mut gate_map=AHashMap::new();
        let mut scale_events=0usize; let mut scale_features=0usize; let mut scale_rows=0usize; let mut scale_vocab=tokens.len();
        for (k,v) in gate_items {
            match k.as_str() {
                "__p89_data_scale_events" => { scale_events = v.max(0.0) as usize; },
                "__p89_data_scale_features" => { scale_features = v.max(0.0) as usize; },
                "__p89_data_scale_rows" => { scale_rows = v.max(0.0) as usize; },
                "__p89_data_scale_vocab" => { scale_vocab = (v.max(0.0) as usize).max(tokens.len()); },
                _ => { gate_map.insert(k, v); },
            }
        }
        let fallback_rows=row_ids.len();
        Self {
            scorer: RustFrozenExactScorer::new(tokens, row_ids, row_ptr, token_ids, values),
            gate_map,
            scale_events,
            scale_features,
            scale_rows: scale_rows.max(fallback_rows),
            scale_vocab,
        }
    }
    fn active_rows_for_text(&self, input_text: String) -> Vec<(u64, f64)> {
        let toks=tokenize_fast(&input_text); let pfeats=prompt_features_scaled(&toks, self.scale_events, self.scale_features, self.scale_rows, self.scale_vocab); self.active_rows(&pfeats, &Vec::new())
    }
    fn predict_text(&self, input_text: String, top_k: usize) -> (String, Vec<(String, f64)>, StdHashMap<String,u64>) {
        let toks=tokenize_fast(&input_text); let copy=copy_tokens(&toks); let pfeats=prompt_features_scaled(&toks, self.scale_events, self.scale_features, self.scale_rows, self.scale_vocab); let rows=self.active_rows(&pfeats, &Vec::new());
        let (row_ids, weights):(Vec<u64>,Vec<f64>)=rows.iter().cloned().unzip();
        let ranked=self.scorer.score_ids_internal(&row_ids, &weights, top_k.max(1));
        let tok_model=ranked.get(0).map(|x| x.0.clone()).unwrap_or_default();
        let tok=self.resolve_copy(&tok_model, &copy);
        let mut d=StdHashMap::new(); d.insert("input_tokens".to_string(), toks.len() as u64); d.insert("prompt_features".to_string(), pfeats.len() as u64); d.insert("active_rows".to_string(), rows.len() as u64); d.insert("ranked".to_string(), ranked.len() as u64);
        (tok, ranked, d)
    }
    fn batch_predict_text(&self, inputs: Vec<String>, top_k: usize) -> Vec<(String, Vec<(String,f64)>)> {
        // P87: parallelize across prompts only.  Each prompt uses a sequential
        // dense/touched scorer to avoid nested Rayon/hash overhead.
        inputs.par_iter().map(|s| { let (tok, ranked, _d)=self.predict_text(s.clone(), top_k); (tok, ranked) }).collect()
    }
    fn batch_generate_text(&self, inputs: Vec<String>, max_tokens: usize, temperature: f64) -> Vec<(String, Vec<String>)> {
        inputs.par_iter().map(|s| { let (text, toks, _d)=self.generate_text(s.clone(), max_tokens, temperature); (text, toks) }).collect()
    }
    fn generate_text(&self, input_text: String, max_tokens: usize, temperature: f64) -> (String, Vec<String>, StdHashMap<String,u64>) {
        let mut cur=input_text.clone(); let mut generated=Vec::<String>::new(); let mut total_rows=0u64;
        let steps=max_tokens;
        for step in 0..steps {
            let toks=tokenize_fast(&cur); let copy=copy_tokens(&toks); let pfeats=prompt_features_scaled(&toks, self.scale_events, self.scale_features, self.scale_rows, self.scale_vocab); let rows=self.active_rows(&pfeats, &generated);
            total_rows += rows.len() as u64;
            let (row_ids, weights):(Vec<u64>,Vec<f64>)=rows.iter().cloned().unzip();
            let ranked = if temperature <= 1e-12 {
                self.scorer.score_ids_internal(&row_ids, &weights, 1)
            } else {
                self.scorer.score_ids_internal(&row_ids, &weights, 0)
            };
            if ranked.is_empty() { break; }
            let chosen=self.choose_token(&ranked, temperature, fnv1a64(&format!("{}|{}", cur, step)));
            let visible=self.resolve_copy(&chosen, &copy);
            if visible.is_empty() || visible == "[EOS]" { break; }
            generated.push(visible.clone());
            if !cur.ends_with(' ') && !cur.ends_with('\n') { cur.push(' '); }
            cur.push_str(&visible);
        }
        let mut d=StdHashMap::new(); d.insert("generated_tokens".to_string(), generated.len() as u64); d.insert("total_active_rows".to_string(), total_rows);
        (detok(&generated), generated, d)
    }
    fn get_statistics(&self) -> StdHashMap<String,u64> {
        let mut m=self.scorer.get_statistics(); m.insert("template_gates".to_string(), self.gate_map.len() as u64); m
    }
}
impl RustFrozenNextEngine {
    fn scale_sum(&self, available: usize, prompt_tokens: usize, active_rows: usize) -> usize {
        (available + prompt_tokens + active_rows + self.scale_events + self.scale_features + self.scale_rows + self.scale_vocab).max(1)
    }
    fn data_feature_budget(&self, available: usize, prompt_tokens: usize, active_rows: usize) -> usize {
        if available == 0 { return 0; }
        let s = self.scale_sum(available, prompt_tokens, active_rows) as f64;
        available.min((s.sqrt().ceil() as usize) + ((s + 1.0).log2().ceil() as usize)).max(1)
    }
    fn take_n(&self, available: usize, prompt_tokens: usize, active_rows: usize) -> usize {
        self.data_feature_budget(available, prompt_tokens, active_rows).min(available)
    }
    fn context_window(&self, prompt_tokens: usize) -> usize {
        let s = (self.scale_events + prompt_tokens + self.scale_vocab).max(1) as f64;
        ((s.sqrt().ceil() as usize) + ((s + 1.0).log2().ceil() as usize)).max(1)
    }
    fn gate(&self, f: &str) -> f64 { self.gate_map.get(&template_id(f)).cloned().unwrap_or(1.0).max(0.0) }
    fn push_row(&self, out: &mut Vec<(u64,f64)>, seen: &mut AHashSet<u64>, f: String, w: f64) {
        if f.is_empty() { return; }
        let rid = fnv1a64(&f);
        if seen.insert(rid) {
            let g = self.gate(&f);
            if g > 0.0 { out.push((rid, w*g)); }
        }
    }
    fn active_rows(&self, pfeats: &[String], prefix: &[String]) -> Vec<(u64,f64)> {
        let sig: Vec<String>=pfeats.iter().filter(|f| f.starts_with("sig:")||f.starts_with("rare2:")||f.starts_with("bag2:")||f.starts_with("pt2:")||f.starts_with("pt3:")).cloned().collect();
        let rare: Vec<String>=pfeats.iter().filter(|f| f.starts_with("rare")||f.starts_with("char:")||f.starts_with("pt:")||f.starts_with("p3:")||f.starts_with("p3e:")||f.starts_with("rpre4:")||f.starts_with("rsuf4:")||f.starts_with("bfrare:")||f.starts_with("bfsk:")).cloned().collect();
        let shape: Vec<String>=pfeats.iter().filter(|f| f.starts_with("pc:")||f.starts_with("prompt:")).cloned().collect();
        let mut out=Vec::<(u64,f64)>::new(); let mut seen=AHashSet::<u64>::new();
        let prompt_tokens = pfeats.len();
        if prefix.is_empty() {
            self.push_row(&mut out,&mut seen,"PX|0".to_string(),1.0);
            let n_sig=self.take_n(sig.len(), prefix.len(), out.len());
            for f in sig.iter().take(n_sig) { self.push_row(&mut out,&mut seen,format!("P0S|{}",f),1.0); }
            let n_rare=self.take_n(rare.len(), prefix.len(), out.len());
            for f in rare.iter().take(n_rare) { self.push_row(&mut out,&mut seen,format!("P0R|{}",f),1.0); }
            let n_shape=self.take_n(shape.len(), prefix.len(), out.len());
            for f in shape.iter().take(n_shape) { self.push_row(&mut out,&mut seen,format!("P0C|{}",f),1.0); }
        } else {
            let l1=prefix[prefix.len()-1].clone(); let c1=tok_class(&l1).to_string();
            self.push_row(&mut out,&mut seen,format!("L1|{}",l1),1.0); self.push_row(&mut out,&mut seen,format!("C1|{}",c1),1.0);
            let n_sig=self.take_n(sig.len(), prefix.len(), out.len());
            for f in sig.iter().take(n_sig) { self.push_row(&mut out,&mut seen,format!("P1S|{}|L1|{}",f,l1),1.0); self.push_row(&mut out,&mut seen,format!("P1SC|{}|C1|{}",f,c1),1.0); }
            let n_rare=self.take_n(rare.len(), prefix.len(), out.len());
            for f in rare.iter().take(n_rare) { self.push_row(&mut out,&mut seen,format!("P1R|{}|L1|{}",f,l1),1.0); }
            if prefix.len()>=2 { let l2=format!("{} {}",prefix[prefix.len()-2],prefix[prefix.len()-1]); let c2=format!("{} {}",tok_class(&prefix[prefix.len()-2]),tok_class(&prefix[prefix.len()-1])); self.push_row(&mut out,&mut seen,format!("L2|{}",l2),1.0); self.push_row(&mut out,&mut seen,format!("C2|{}",c2),1.0); let n_sig2=self.take_n(sig.len(), prefix.len(), out.len()); for f in sig.iter().take(n_sig2) { self.push_row(&mut out,&mut seen,format!("P2S|{}|L2|{}",f,l2),1.0); } let n_rare2=self.take_n(rare.len(), prefix.len(), out.len()); for f in rare.iter().take(n_rare2) { self.push_row(&mut out,&mut seen,format!("P2R|{}|L2|{}",f,l2),1.0); } }
            if prefix.len()>=3 { let l3=format!("{} {} {}",prefix[prefix.len()-3],prefix[prefix.len()-2],prefix[prefix.len()-1]); let c3=format!("{} {} {}",tok_class(&prefix[prefix.len()-3]),tok_class(&prefix[prefix.len()-2]),tok_class(&prefix[prefix.len()-1])); self.push_row(&mut out,&mut seen,format!("L3|{}",l3),1.0); self.push_row(&mut out,&mut seen,format!("C3|{}",c3),1.0); let n_sig3=self.take_n(sig.len(), prefix.len(), out.len()); for f in sig.iter().take(n_sig3) { self.push_row(&mut out,&mut seen,format!("P3S|{}|L3|{}",f,l3),1.0); } let n_rare3=self.take_n(rare.len(), prefix.len(), out.len()); for f in rare.iter().take(n_rare3) { self.push_row(&mut out,&mut seen,format!("P3R|{}|L3|{}",f,l3),1.0); } }
        }
        let n_pb=self.take_n(sig.len(), prefix.len(), out.len());
        for f in sig.iter().take(n_pb) { self.push_row(&mut out,&mut seen,format!("PB|{}",f),1.0); }
        if !prefix.is_empty() {
            let win=self.context_window(prompt_tokens).min(prefix.len());
            let tail: Vec<String>=prefix[prefix.len()-win..].to_vec();
            let pos_budget=self.data_feature_budget(tail.len(), prefix.len(), out.len());
            for (j,tok) in tail.iter().rev().take(pos_budget).enumerate() { self.push_row(&mut out,&mut seen,format!("R{}|{}",j,tok),1.0); self.push_row(&mut out,&mut seen,format!("RC{}|{}",j,tok_class(tok)),1.0); }
            let mut bag=AHashSet::<String>::new(); let bag_budget=self.data_feature_budget(tail.len(), prefix.len(), out.len());
            for tok in tail.iter().rev() { if !bag.contains(tok) { bag.insert(tok.clone()); self.push_row(&mut out,&mut seen,format!("PBAG|{}",tok),1.0); if bag.len()>=bag_budget { break; } } }
        }
        self.push_row(&mut out,&mut seen,format!("LEN|{}",prefix.len()),1.0); let lenlog=((prefix.len()+1) as f64).log2().ceil() as usize; self.push_row(&mut out,&mut seen,format!("LENLOG|{}",lenlog),1.0);
        out
    }
    fn resolve_copy(&self, tok: &str, copy: &[String]) -> String { if tok.starts_with("[COPY") && tok.ends_with(']') { let num=&tok[5..tok.len()-1]; if let Ok(i)=num.parse::<usize>() { if i<copy.len() { return copy[i].clone(); } } } tok.to_string() }
    fn choose_token(&self, ranked: &[(String,f64)], temp: f64, seed: u64) -> String { if ranked.is_empty() { return String::new(); } if temp <= 1e-12 { return ranked[0].0.clone(); } let maxv=ranked.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max); let mut vals=Vec::<f64>::new(); let mut total=0.0; for (_t,s) in ranked { let v=((s-maxv)/temp.max(1e-9)).exp(); vals.push(v); total+=v; } let r=(splitmix64(seed) as f64 / u64::MAX as f64) * total; let mut acc=0.0; for (i,v) in vals.iter().enumerate() { acc+=*v; if acc>=r { return ranked[i].0.clone(); } } ranked[0].0.clone() }
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustHebbianHdcCountEvidence>()?;
    m.add_class::<RustHdcCountEvidence>()?;
    m.add_class::<RustCountEvidence>()?;
    m.add_class::<RustLogitTables>()?;
    m.add_class::<RustFrozenExactScorer>()?;
    m.add_class::<RustFrozenNextEngine>()?;
    Ok(())
}
