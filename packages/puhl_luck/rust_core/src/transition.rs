/// Transition Memory Fast Path — pyo3 0.23
///
/// Accelerates the hot paths in TransitionMemoryLayer:
/// - find_similar_transitions_by_features: O(n) feature-overlap search in parallel
/// - store_transition_features: compute completion features fast
/// - batch_transition_similarity: for operator induction clustering

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rayon::prelude::*;

/// Find top-k transitions by feature-overlap similarity (Jaccard).
///
/// Args:
///     query_features: features of the current partial state
///     stored: list of (transition_id, partial_features, complete_features, relevance_count)
///     top_k: max results
///     min_sim: minimum similarity threshold
///
/// Returns:
///     [(transition_id, similarity), ...] sorted descending
#[pyfunction]
#[pyo3(signature = (query_features, stored, top_k=10, min_sim=0.0))]
pub fn find_transitions_by_features_rust(
    py: Python,
    query_features: Vec<String>,
    stored: Vec<(String, Vec<String>, Vec<String>, i64)>,
    top_k: usize,
    min_sim: f64,
) -> PyResult<PyObject> {
    let qset: std::collections::HashSet<&str> = query_features.iter().map(|s| s.as_str()).collect();

    let mut results: Vec<(String, f64)> = stored.par_iter().filter_map(|(tid, partial, _complete, relevance)| {
        let pset: std::collections::HashSet<&str> = partial.iter().map(|s| s.as_str()).collect();
        let inter = qset.intersection(&pset).count();
        let union = qset.union(&pset).count();
        if union == 0 { return None; }
        let sim = inter as f64 / union as f64;
        // Boost by log(relevance+1) to surface frequently-used transitions
        let boosted = sim * (1.0 + (*relevance as f64).ln_1p() * 0.1);
        if boosted >= min_sim { Some((tid.clone(), boosted)) } else { None }
    }).collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);

    let list = PyList::empty(py);
    for (tid, sim) in results {
        list.append(PyTuple::new(py, [tid.into_pyobject(py)?.into_any(), sim.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

/// Compute completion features (added in complete but not in partial) for many transitions.
///
/// Args:
///     pairs: list of (transition_id, partial_features, complete_features)
///
/// Returns:
///     [(transition_id, [added_features]), ...]
#[pyfunction]
pub fn batch_completion_features_rust(
    py: Python,
    pairs: Vec<(String, Vec<String>, Vec<String>)>,
) -> PyResult<PyObject> {
    let results: Vec<(String, Vec<String>)> = pairs.par_iter().map(|(tid, partial, complete)| {
        let ps: std::collections::HashSet<&str> = partial.iter().map(|s| s.as_str()).collect();
        let added: Vec<String> = complete.iter()
            .filter(|f| !ps.contains(f.as_str()))
            .cloned()
            .collect();
        (tid.clone(), added)
    }).collect();
    let list = PyList::empty(py);
    for (tid, added) in results {
        list.append(PyTuple::new(py, [tid.into_pyobject(py)?.into_any(), added.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

/// Compute pairwise feature-overlap similarities for operator induction clustering.
///
/// Args:
///     items: list of (id, features)
///
/// Returns:
///     [(id_a, id_b, similarity), ...] for all pairs with sim > 0
#[pyfunction]
#[pyo3(signature = (items, min_sim=0.0))]
pub fn pairwise_feature_similarity_rust(
    py: Python,
    items: Vec<(String, Vec<String>)>,
    min_sim: f64,
) -> PyResult<PyObject> {
    let n = items.len();
    // Collect all pairs in parallel using flat_map
    let pairs: Vec<(String, String, f64)> = (0..n).into_par_iter().flat_map(|i| {
        let (ia, fa) = &items[i];
        let sa: std::collections::HashSet<&str> = fa.iter().map(|s| s.as_str()).collect();
        let mut row = Vec::new();
        for j in (i+1)..n {
            let (ib, fb) = &items[j];
            let sb: std::collections::HashSet<&str> = fb.iter().map(|s| s.as_str()).collect();
            let inter = sa.intersection(&sb).count();
            let union = sa.union(&sb).count();
            let sim = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
            if sim >= min_sim {
                row.push((ia.clone(), ib.clone(), sim));
            }
        }
        row
    }).collect();
    let list = PyList::empty(py);
    for (ia, ib, sim) in pairs {
        list.append(PyTuple::new(py, [ia.into_pyobject(py)?.into_any(), ib.into_pyobject(py)?.into_any(), sim.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_transitions_by_features_rust, m)?)?;
    m.add_function(wrap_pyfunction!(batch_completion_features_rust, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_feature_similarity_rust, m)?)?;
    Ok(())
}
