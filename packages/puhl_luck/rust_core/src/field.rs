/// Field Operations — pyo3 0.23

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (items, query_features, base_activation=1.0))]
fn activate_field_items_rust(
    py: Python,
    items: Vec<(String, Vec<String>)>,
    query_features: Vec<String>,
    base_activation: f64,
) -> PyResult<PyObject> {
    let qset: std::collections::HashSet<&str> = query_features.iter().map(|s| s.as_str()).collect();
    let activations: Vec<(String, f64)> = items.par_iter().filter_map(|(id, feats)| {
        let fset: std::collections::HashSet<&str> = feats.iter().map(|s| s.as_str()).collect();
        let inter = qset.intersection(&fset).count();
        let union = qset.union(&fset).count();
        let sim = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
        let a = base_activation * sim;
        if a > 0.0 { Some((id.clone(), a)) } else { None }
    }).collect();
    let dict = PyDict::new(py);
    for (id, a) in activations { dict.set_item(id, a)?; }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (current_activations, evidence_items, conflict_items, evidence_boost=1.2, conflict_damping=0.8))]
fn update_field_activations_rust(
    py: Python,
    current_activations: HashMap<String, f64>,
    evidence_items: Vec<String>,
    conflict_items: Vec<String>,
    evidence_boost: f64,
    conflict_damping: f64,
) -> PyResult<PyObject> {
    let mut updated = current_activations;
    for id in &evidence_items { if let Some(a) = updated.get_mut(id) { *a *= evidence_boost; } }
    for id in &conflict_items { if let Some(a) = updated.get_mut(id) { *a *= conflict_damping; } }
    let max = updated.values().cloned().fold(0.0f64, f64::max);
    if max > 0.0 { for a in updated.values_mut() { *a /= max; } }
    let dict = PyDict::new(py);
    for (k, v) in updated { dict.set_item(k, v)?; }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (items, activations, min_activation=0.1))]
fn compute_pairwise_resonance_rust(
    py: Python,
    items: Vec<(String, Vec<String>)>,
    activations: HashMap<String, f64>,
    min_activation: f64,
) -> PyResult<PyObject> {
    let active: Vec<_> = items.iter()
        .filter(|(id, _)| activations.get(id).copied().unwrap_or(0.0) >= min_activation)
        .collect();
    let dict = PyDict::new(py);
    for i in 0..active.len() {
        for j in (i+1)..active.len() {
            let (ia, fa) = active[i];
            let (ib, fb) = active[j];
            let sa: std::collections::HashSet<&str> = fa.iter().map(|s| s.as_str()).collect();
            let sb: std::collections::HashSet<&str> = fb.iter().map(|s| s.as_str()).collect();
            let inter = sa.intersection(&sb).count();
            let union = sa.union(&sb).count();
            let r = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
            if r > 0.0 {
                let key = PyTuple::new(py, [ia.as_str(), ib.as_str()])?;
                dict.set_item(key, r)?;
            }
        }
    }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (activations, k=10, min_threshold=0.0))]
fn get_top_k_activations_rust(
    py: Python,
    activations: HashMap<String, f64>,
    k: usize,
    min_threshold: f64,
) -> PyResult<PyObject> {
    let mut items: Vec<_> = activations.into_iter().filter(|(_, a)| *a >= min_threshold).collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    items.truncate(k);
    let list = PyList::empty(py);
    for (id, a) in items {
        list.append(PyTuple::new(py, [id.into_pyobject(py)?.into_any(), a.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

#[pyfunction]
#[pyo3(signature = (fields, merge_strategy="max"))]
fn merge_field_states_rust(
    py: Python,
    fields: Vec<HashMap<String, f64>>,
    merge_strategy: &str,
) -> PyResult<PyObject> {
    let mut merged: HashMap<String, Vec<f64>> = HashMap::new();
    for field in fields {
        for (id, a) in field { merged.entry(id).or_default().push(a); }
    }
    let result: HashMap<String, f64> = merged.into_iter().map(|(id, vals)| {
        let v = match merge_strategy {
            "sum" => vals.iter().sum(),
            "average" => vals.iter().sum::<f64>() / vals.len() as f64,
            _ => vals.iter().cloned().fold(0.0f64, f64::max),
        };
        (id, v)
    }).collect();
    let dict = PyDict::new(py);
    for (k, v) in result { dict.set_item(k, v)?; }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (previous_activations, current_activations, threshold=0.01))]
fn detect_field_convergence_rust(
    previous_activations: HashMap<String, f64>,
    current_activations: HashMap<String, f64>,
    threshold: f64,
) -> (bool, f64) {
    let mut max_delta = 0.0f64;
    for (id, curr) in &current_activations {
        let prev = previous_activations.get(id).copied().unwrap_or(0.0);
        max_delta = max_delta.max((curr - prev).abs());
    }
    for (id, prev) in &previous_activations {
        if !current_activations.contains_key(id) { max_delta = max_delta.max(*prev); }
    }
    (max_delta <= threshold, max_delta)
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(activate_field_items_rust, m)?)?;
    m.add_function(wrap_pyfunction!(update_field_activations_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pairwise_resonance_rust, m)?)?;
    m.add_function(wrap_pyfunction!(get_top_k_activations_rust, m)?)?;
    m.add_function(wrap_pyfunction!(merge_field_states_rust, m)?)?;
    m.add_function(wrap_pyfunction!(detect_field_convergence_rust, m)?)?;
    Ok(())
}
