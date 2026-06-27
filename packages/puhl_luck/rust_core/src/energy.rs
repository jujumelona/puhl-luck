/// Field Energy Computation — pyo3 0.23

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (conflicts, evidence, conflict_penalty=1.0, evidence_boost=1.0))]
fn compute_field_energy_rust(
    conflicts: Vec<(String, String, f64)>,
    evidence: Vec<(String, f64)>,
    conflict_penalty: f64,
    evidence_boost: f64,
) -> f64 {
    let c: f64 = conflicts.par_iter().map(|(_, _, w)| w * conflict_penalty).sum();
    let e: f64 = evidence.par_iter().map(|(_, w)| w * evidence_boost).sum();
    c - e
}

#[pyfunction]
#[pyo3(signature = (current_energy, candidate_id, new_conflicts, new_evidence, conflict_penalty=1.0, evidence_boost=1.0))]
fn compute_energy_delta_rust(
    current_energy: f64,
    candidate_id: &str,
    new_conflicts: Vec<(String, String, f64)>,
    new_evidence: Vec<(String, f64)>,
    conflict_penalty: f64,
    evidence_boost: f64,
) -> f64 {
    let _ = (current_energy, candidate_id);
    let c: f64 = new_conflicts.par_iter().map(|(_, _, w)| w * conflict_penalty).sum();
    let e: f64 = new_evidence.par_iter().map(|(_, w)| w * evidence_boost).sum();
    c - e
}

#[pyfunction]
#[pyo3(signature = (conflicts, evidence, conflict_penalty=1.0, evidence_boost=1.0))]
fn compute_energy_by_source_rust(
    py: Python,
    conflicts: Vec<(String, String, f64)>,
    evidence: Vec<(String, f64)>,
    conflict_penalty: f64,
    evidence_boost: f64,
) -> PyResult<PyObject> {
    let mut map: HashMap<String, f64> = HashMap::new();
    for (src, _, w) in conflicts { *map.entry(src).or_insert(0.0) += w * conflict_penalty; }
    for (id, w) in evidence { *map.entry(id).or_insert(0.0) -= w * evidence_boost; }
    let dict = PyDict::new(py);
    for (k, v) in map { dict.set_item(k, v)?; }
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (base_energy, candidates, conflict_penalty=1.0, evidence_boost=1.0))]
fn batch_compute_energy_rust(
    py: Python,
    base_energy: f64,
    candidates: Vec<(String, Vec<(String, String, f64)>, Vec<(String, f64)>)>,
    conflict_penalty: f64,
    evidence_boost: f64,
) -> PyResult<PyObject> {
    let _ = base_energy;
    let results: Vec<(String, f64)> = candidates.par_iter()
        .map(|(cid, conflicts, evidence)| {
            let c: f64 = conflicts.iter().map(|(_, _, w)| w * conflict_penalty).sum();
            let e: f64 = evidence.iter().map(|(_, w)| w * evidence_boost).sum();
            (cid.clone(), c - e)
        })
        .collect();
    let list = PyList::empty(py);
    for (cid, delta) in results {
        list.append(PyTuple::new(py, [cid.into_pyobject(py)?.into_any(), delta.into_pyobject(py)?.into_any()])?)?;
    }
    Ok(list.into())
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_field_energy_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_energy_delta_rust, m)?)?;
    m.add_function(wrap_pyfunction!(compute_energy_by_source_rust, m)?)?;
    m.add_function(wrap_pyfunction!(batch_compute_energy_rust, m)?)?;
    Ok(())
}
