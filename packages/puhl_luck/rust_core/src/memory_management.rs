/// Memory Management — pyo3 0.23

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[pyfunction]
pub fn prune_events_rust(
    event_ids: Vec<String>,
    event_novelty: HashMap<String, f64>,
    event_last_accessed: HashMap<String, i64>,
    event_activation: HashMap<String, f64>,
    protected_ids: HashSet<String>,
    target_count: usize,
    current_time: i64,
) -> Vec<String> {
    if event_ids.len() <= target_count { return Vec::new(); }
    let num_to_prune = event_ids.len() - target_count;
    let mut scored: Vec<(String, f64)> = event_ids.par_iter().filter_map(|id| {
        if protected_ids.contains(id) { return None; }
        let novelty = event_novelty.get(id).copied().unwrap_or(0.0);
        let last = event_last_accessed.get(id).copied().unwrap_or(0);
        let activation = event_activation.get(id).copied().unwrap_or(0.0);
        let days = ((current_time - last) / 86400).max(1) as f64;
        let score = 0.3 * novelty + 0.3 / days + 0.4 * activation;
        Some((id.clone(), score))
    }).collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.iter().take(num_to_prune).map(|(id, _)| id.clone()).collect()
}

#[pyfunction]
pub fn prune_operators_rust(
    operator_ids: Vec<String>,
    operator_confidence: HashMap<String, f64>,
    operator_use_count: HashMap<String, i64>,
    operator_success_count: HashMap<String, i64>,
    operator_last_used: HashMap<String, i64>,
    min_confidence: f64,
    min_use_count: i64,
    current_time: i64,
) -> Vec<String> {
    operator_ids.par_iter().filter_map(|id| {
        let conf = operator_confidence.get(id).copied().unwrap_or(0.0);
        let use_c = operator_use_count.get(id).copied().unwrap_or(0);
        let succ = operator_success_count.get(id).copied().unwrap_or(0);
        let last = operator_last_used.get(id).copied().unwrap_or(0);
        let rate = if use_c > 0 { succ as f64 / use_c as f64 } else { 0.0 };
        let days = ((current_time - last) / 86400).max(1) as f64;
        let prune = (conf < min_confidence && use_c < min_use_count)
            || (days > 180.0 && rate < 0.3)
            || (use_c == 0 && conf < min_confidence * 1.5);
        if prune { Some(id.clone()) } else { None }
    }).collect()
}

#[pyfunction]
pub fn prune_transitions_rust(
    transition_ids: Vec<String>,
    transition_relevance: HashMap<String, f64>,
    transition_match_count: HashMap<String, i64>,
    transition_last_matched: HashMap<String, i64>,
    target_count: usize,
    current_time: i64,
) -> Vec<String> {
    if transition_ids.len() <= target_count { return Vec::new(); }
    let num_to_prune = transition_ids.len() - target_count;
    let mut scored: Vec<(String, f64)> = transition_ids.par_iter().map(|id| {
        let rel = transition_relevance.get(id).copied().unwrap_or(0.0);
        let mc = transition_match_count.get(id).copied().unwrap_or(0);
        let last = transition_last_matched.get(id).copied().unwrap_or(0);
        let days = ((current_time - last) / 86400).max(1) as f64;
        let score = 0.4 * rel + 0.3 / days.sqrt() + 0.3 * (mc as f64 + 1.0).ln();
        (id.clone(), score)
    }).collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.iter().take(num_to_prune).map(|(id, _)| id.clone()).collect()
}

#[pyfunction]
pub fn compute_memory_health_rust(
    py: Python,
    event_count: usize,
    operator_count: usize,
    transition_count: usize,
    event_novelty_values: Vec<f64>,
    operator_confidence_values: Vec<f64>,
    transition_relevance_values: Vec<f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("event_count", event_count)?;
    dict.set_item("operator_count", operator_count)?;
    dict.set_item("transition_count", transition_count)?;
    dict.set_item("total_memory_items", event_count + operator_count + transition_count)?;
    let avg = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    dict.set_item("avg_event_novelty", avg(&event_novelty_values))?;
    dict.set_item("avg_operator_confidence", avg(&operator_confidence_values))?;
    dict.set_item("avg_transition_relevance", avg(&transition_relevance_values))?;
    let health = (if event_count < 10000 { 1.0 } else { 10000.0 / event_count as f64 }) * 0.4
        + avg(&operator_confidence_values) * 0.3
        + avg(&transition_relevance_values) * 0.3;
    dict.set_item("memory_health_score", health)?;
    dict.set_item("needs_pruning", event_count > 10000 || operator_count > 1000 || transition_count > 5000)?;
    Ok(dict.into())
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prune_events_rust, m)?)?;
    m.add_function(wrap_pyfunction!(prune_operators_rust, m)?)?;
    m.add_function(wrap_pyfunction!(prune_transitions_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_memory_health_rust, m)?)?;
    Ok(())
}
