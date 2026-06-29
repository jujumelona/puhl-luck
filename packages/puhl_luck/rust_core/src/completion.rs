/// State Completion — pyo3 0.23

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::collections::{HashMap, HashSet};

#[pyfunction]
#[pyo3(signature = (incomplete_state, context_features, completion_patterns, max_candidates=5))]
pub fn complete_state_rust(
    py: Python,
    incomplete_state: Vec<String>,
    context_features: Vec<String>,
    completion_patterns: Vec<(Vec<String>, Vec<String>, f64)>,
    max_candidates: usize,
) -> PyResult<PyObject> {
    let incomplete: HashSet<&str> = incomplete_state.iter().map(|s| s.as_str()).collect();
    let context: HashSet<&str> = context_features.iter().map(|s| s.as_str()).collect();

    let mut scored: Vec<(Vec<String>, f64)> = completion_patterns.iter()
        .filter_map(|(pattern, completion, base_conf)| {
            let pset: HashSet<&str> = pattern.iter().map(|s| s.as_str()).collect();
            if pset.is_empty() { return None; }
            let state_match = incomplete.intersection(&pset).count() as f64 / pset.len() as f64;
            let ctx_match = context.intersection(&pset).count() as f64 / pset.len() as f64;
            let score = base_conf * (0.7 * state_match + 0.3 * ctx_match);
            if score > 0.1 { Some((completion.clone(), score)) } else { None }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(max_candidates);

    let list = PyList::empty(py);
    for (completion, confidence) in scored {
        list.append(PyTuple::new(py, [
            completion.into_pyobject(py)?.into_any(),
            confidence.into_pyobject(py)?.into_any(),
        ])?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (candidates, merge_strategy="weighted"))]
fn merge_completions_rust(
    py: Python,
    candidates: Vec<(Vec<String>, f64)>,
    merge_strategy: &str,
) -> PyResult<PyObject> {
    if candidates.is_empty() { return Ok(PyList::empty(py).into()); }
    let result: Vec<String> = match merge_strategy {
        "union" => {
            let mut all: HashSet<String> = HashSet::new();
            for (f, _) in &candidates { all.extend(f.iter().cloned()); }
            all.into_iter().collect()
        }
        "intersection" => {
            let sets: Vec<HashSet<String>> = candidates.iter()
                .map(|(f, _)| f.iter().cloned().collect())
                .collect();
            if sets.is_empty() { return Ok(PyList::empty(py).into()); }
            let mut inter = sets[0].clone();
            for s in &sets[1..] { inter = inter.intersection(s).cloned().collect(); }
            inter.into_iter().collect()
        }
        _ => {
            let total: f64 = candidates.iter().map(|(_, c)| c).sum();
            let mut scores: HashMap<String, f64> = HashMap::new();
            for (feats, conf) in &candidates {
                let norm = conf / total;
                for f in feats { *scores.entry(f.clone()).or_insert(0.0) += norm; }
            }
            let mut weighted: Vec<(String, f64)> = scores.into_iter().collect();
            weighted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            weighted.into_iter().filter(|(_, s)| *s > 0.3).map(|(f, _)| f).collect()
        }
    };
    let list = PyList::empty(py);
    for f in result { list.append(f)?; }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (completions, context, quality_metrics=None))]
fn rank_completions_rust(
    py: Python,
    completions: Vec<Vec<String>>,
    context: Vec<String>,
    quality_metrics: Option<HashMap<String, f64>>,
) -> PyResult<PyObject> {
    let metrics = quality_metrics.unwrap_or_else(|| {
        [("coherence".to_string(), 0.4), ("completeness".to_string(), 0.3), ("relevance".to_string(), 0.3)]
            .into_iter().collect()
    });
    let ctx: HashSet<&str> = context.iter().map(|s| s.as_str()).collect();
    let mut scored: Vec<(Vec<String>, f64)> = completions.into_iter().map(|c| {
        let cset: HashSet<&str> = c.iter().map(|s| s.as_str()).collect();
        let coherence = 1.0 - (c.len() - cset.len()) as f64 / c.len().max(1) as f64;
        let completeness = (c.len() as f64 / 10.0).min(1.0);
        let relevance = ctx.intersection(&cset).count() as f64 / ctx.len().max(1) as f64;
        let q = metrics.get("coherence").unwrap_or(&0.4) * coherence
            + metrics.get("completeness").unwrap_or(&0.3) * completeness
            + metrics.get("relevance").unwrap_or(&0.3) * relevance;
        (c, q)
    }).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let list = PyList::empty(py);
    for (c, q) in scored {
        list.append(PyTuple::new(py, [c.into_pyobject(py)?.into_any(), q.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(complete_state_rust, m)?)?;
    m.add_function(wrap_pyfunction!(merge_completions_rust, m)?)?;
    m.add_function(wrap_pyfunction!(rank_completions_rust, m)?)?;
    Ok(())
}
