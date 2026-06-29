/// Pattern Matching and Search — pyo3 0.23

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (query_features, items, top_k=10, min_similarity=0.0))]
fn find_similar_items_rust(
    py: Python,
    query_features: Vec<String>,
    items: Vec<(String, Vec<String>, PyObject)>,
    top_k: usize,
    min_similarity: f64,
) -> PyResult<PyObject> {
    let qset: std::collections::HashSet<&str> = query_features.iter().map(|s| s.as_str()).collect();
    let mut results: Vec<(String, f64, &PyObject)> = items.iter().filter_map(|(id, feats, meta)| {
        let fset: std::collections::HashSet<&str> = feats.iter().map(|s| s.as_str()).collect();
        let inter = qset.intersection(&fset).count();
        let union = qset.union(&fset).count();
        let sim = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
        if sim >= min_similarity { Some((id.clone(), sim, meta)) } else { None }
    }).collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(top_k);
    let list = PyList::empty(py);
    for (id, sim, meta) in results {
        list.append(PyTuple::new(py, [id.into_pyobject(py)?.into_any(), sim.into_pyobject(py)?.into_any(), meta.clone_ref(py).into_bound(py).into_any()])?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (sequence, min_pattern_length=2, max_pattern_length=5, min_occurrences=2))]
fn extract_repeated_patterns_rust(
    py: Python,
    sequence: Vec<String>,
    min_pattern_length: usize,
    max_pattern_length: usize,
    min_occurrences: usize,
) -> PyResult<PyObject> {
    let n = sequence.len();
    let mut counts: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
    for len in min_pattern_length..=max_pattern_length.min(n) {
        for start in 0..=(n - len) {
            let pat = sequence[start..start+len].to_vec();
            counts.entry(pat).or_default().push(start);
        }
    }
    let mut results: Vec<_> = counts.into_iter()
        .filter(|(_, pos)| pos.len() >= min_occurrences)
        .map(|(pat, pos)| (pat, pos.len(), pos))
        .collect();
    results.sort_by(|a, b| b.1.cmp(&a.1));
    let list = PyList::empty(py);
    for (pat, count, pos) in results {
        list.append(PyTuple::new(py, [pat.into_pyobject(py)?.into_any(), count.into_pyobject(py)?.into_any(), pos.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (current_state, operators, min_match=0.5))]
pub fn find_applicable_operators_rust(
    py: Python,
    current_state: Vec<String>,
    operators: Vec<(String, Vec<String>, Vec<String>, f64)>,
    min_match: f64,
) -> PyResult<PyObject> {
    let state: std::collections::HashSet<&str> = current_state.iter().map(|s| s.as_str()).collect();
    let mut results: Vec<(String, f64, Vec<String>, f64)> = operators.into_iter().filter_map(|(id, preconds, effects, conf)| {
        if preconds.is_empty() { return None; }
        let ps: std::collections::HashSet<&str> = preconds.iter().map(|s| s.as_str()).collect();
        let matched = state.intersection(&ps).count();
        let ratio = matched as f64 / preconds.len() as f64;
        if ratio >= min_match { Some((id, ratio, effects, conf)) } else { None }
    }).collect();
    results.sort_by(|a, b| (b.1 * b.3).partial_cmp(&(a.1 * a.3)).unwrap());
    let list = PyList::empty(py);
    for (id, score, effects, conf) in results {
        list.append(PyTuple::new(py, [id.into_pyobject(py)?.into_any(), score.into_pyobject(py)?.into_any(), effects.into_pyobject(py)?.into_any(), conf.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (partial_state, transitions, top_k=5, min_similarity=0.3))]
pub fn find_similar_transitions_rust(
    py: Python,
    partial_state: Vec<String>,
    transitions: Vec<(String, Vec<String>, Vec<String>, f64)>,
    top_k: usize,
    min_similarity: f64,
) -> PyResult<PyObject> {
    let pset: std::collections::HashSet<&str> = partial_state.iter().map(|s| s.as_str()).collect();
    let mut results: Vec<(String, f64, Vec<String>, f64)> = transitions.par_iter().filter_map(|(tid, from, to, conf)| {
        let fset: std::collections::HashSet<&str> = from.iter().map(|s| s.as_str()).collect();
        let inter = pset.intersection(&fset).count();
        let union = pset.union(&fset).count();
        let sim = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
        if sim >= min_similarity { Some((tid.clone(), sim, to.clone(), *conf)) } else { None }
    }).collect();
    results.sort_by(|a, b| (b.1 * b.3).partial_cmp(&(a.1 * a.3)).unwrap());
    results.truncate(top_k);
    let list = PyList::empty(py);
    for (tid, sim, to, conf) in results {
        list.append(PyTuple::new(py, [tid.into_pyobject(py)?.into_any(), sim.into_pyobject(py)?.into_any(), to.into_pyobject(py)?.into_any(), conf.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (queries, items, min_similarity=0.1))]
fn batch_similarity_search_rust(
    py: Python,
    queries: Vec<(String, Vec<String>)>,
    items: Vec<(String, Vec<String>)>,
    min_similarity: f64,
) -> PyResult<PyObject> {
    let results: Vec<(String, Vec<(String, f64)>)> = queries.par_iter().map(|(qid, qfeats)| {
        let qset: std::collections::HashSet<&str> = qfeats.iter().map(|s| s.as_str()).collect();
        let mut matches: Vec<(String, f64)> = items.iter().filter_map(|(iid, ifeats)| {
            let iset: std::collections::HashSet<&str> = ifeats.iter().map(|s| s.as_str()).collect();
            let inter = qset.intersection(&iset).count();
            let union = qset.union(&iset).count();
            let sim = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
            if sim >= min_similarity { Some((iid.clone(), sim)) } else { None }
        }).collect();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        (qid.clone(), matches)
    }).collect();
    let dict = PyDict::new(py);
    for (qid, matches) in results {
        let ml = PyList::empty(py);
        for (iid, sim) in matches {
            ml.append(PyTuple::new(py, [iid.into_pyobject(py)?.into_any(), sim.into_pyobject(py)?.into_any()])?)?;
        }
        dict.set_item(qid, ml)?;
    }
    Ok(dict.into())
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_similar_items_rust, m)?)?;
    m.add_function(wrap_pyfunction!(extract_repeated_patterns_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_applicable_operators_rust, m)?)?;
    m.add_function(wrap_pyfunction!(find_similar_transitions_rust, m)?)?;
    m.add_function(wrap_pyfunction!(batch_similarity_search_rust, m)?)?;
    Ok(())
}
