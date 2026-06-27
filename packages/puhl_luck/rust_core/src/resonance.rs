/// Resonance Calculation — pyo3 0.23

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (item_a_features, item_b_features, coactivation_count=0, has_conflict=false, conflict_penalty=-1.0))]
pub fn compute_resonance_rust(
    item_a_features: Vec<String>,
    item_b_features: Vec<String>,
    coactivation_count: i32,
    has_conflict: bool,
    conflict_penalty: f64,
) -> f64 {
    let a: std::collections::HashSet<&str> = item_a_features.iter().map(|s| s.as_str()).collect();
    let b: std::collections::HashSet<&str> = item_b_features.iter().map(|s| s.as_str()).collect();
    let inter = a.intersection(&b).count();
    let union = a.union(&b).count();
    let overlap = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
    let coact = (coactivation_count as f64).ln_1p() * 0.1;
    let conflict = if has_conflict { conflict_penalty } else { 0.0 };
    overlap + coact + conflict
}

#[pyfunction]
#[pyo3(signature = (similarity, coactivation_count=0, has_conflict=false, conflict_penalty=-1.0))]
fn compute_resonance_from_similarity_rust(
    similarity: f64, coactivation_count: i32, has_conflict: bool, conflict_penalty: f64,
) -> f64 {
    let coact = (coactivation_count as f64).ln_1p() * 0.1;
    let conflict = if has_conflict { conflict_penalty } else { 0.0 };
    similarity + coact + conflict
}

#[pyfunction]
#[pyo3(signature = (query_features, targets, conflict_penalty=-1.0))]
fn batch_compute_resonance_rust(
    py: Python,
    query_features: Vec<String>,
    targets: Vec<(String, Vec<String>, i32, bool)>,
    conflict_penalty: f64,
) -> PyResult<PyObject> {
    let qset: std::collections::HashSet<&str> = query_features.iter().map(|s| s.as_str()).collect();
    let results: Vec<(String, f64)> = targets.par_iter().map(|(tid, tfeats, coact, conflict)| {
        let tset: std::collections::HashSet<&str> = tfeats.iter().map(|s| s.as_str()).collect();
        let inter = qset.intersection(&tset).count();
        let union = qset.union(&tset).count();
        let overlap = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
        let ca = (*coact as f64).ln_1p() * 0.1;
        let cf = if *conflict { conflict_penalty } else { 0.0 };
        (tid.clone(), overlap + ca + cf)
    }).collect();
    let list = PyList::empty(py);
    for (tid, r) in results {
        list.append(PyTuple::new(py, [tid.into_pyobject(py)?.into_any(), r.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (initial_activations, resonance_matrix, iterations=3, damping=0.85))]
fn propagate_resonance_rust(
    py: Python,
    initial_activations: HashMap<String, f64>,
    resonance_matrix: HashMap<(String, String), f64>,
    iterations: usize,
    damping: f64,
) -> PyResult<PyObject> {
    let mut act = initial_activations;
    for _ in 0..iterations {
        let mut next: HashMap<String, f64> = HashMap::new();
        for (id, &a) in &act {
            let mut boost = 0.0f64;
            for (oid, &oa) in &act {
                if id != oid {
                    let r = resonance_matrix.get(&(id.clone(), oid.clone()))
                        .or_else(|| resonance_matrix.get(&(oid.clone(), id.clone())))
                        .copied().unwrap_or(0.0);
                    boost += r * oa;
                }
            }
            next.insert(id.clone(), (a * (1.0 - damping) + boost * damping).max(0.0));
        }
        act = next;
    }
    let dict = PyDict::new(py);
    for (k, v) in act { dict.set_item(k, v)?; }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (activations, resonance_matrix, threshold=0.5))]
fn identify_resonance_clusters_rust(
    py: Python,
    activations: HashMap<String, f64>,
    resonance_matrix: HashMap<(String, String), f64>,
    threshold: f64,
) -> PyResult<PyObject> {
    let items: Vec<String> = activations.keys().cloned().collect();
    let mut clusters: Vec<Vec<String>> = Vec::new();
    let mut assigned = std::collections::HashSet::new();
    for item in &items {
        if assigned.contains(item) { continue; }
        let mut cluster = vec![item.clone()];
        assigned.insert(item.clone());
        for other in &items {
            if assigned.contains(other) { continue; }
            let r = resonance_matrix.get(&(item.clone(), other.clone()))
                .or_else(|| resonance_matrix.get(&(other.clone(), item.clone())))
                .copied().unwrap_or(0.0);
            if r >= threshold { cluster.push(other.clone()); assigned.insert(other.clone()); }
        }
        clusters.push(cluster);
    }
    let list = PyList::empty(py);
    for cluster in clusters {
        let cl = PyList::empty(py);
        for id in cluster { cl.append(id)?; }
        list.append(cl)?;
    }
    Ok(list.into())
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_resonance_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_resonance_from_similarity_rust, m)?)?;
    m.add_function(wrap_pyfunction!(batch_compute_resonance_rust, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_resonance_rust, m)?)?;
    m.add_function(wrap_pyfunction!(identify_resonance_clusters_rust, m)?)?;
    Ok(())
}
