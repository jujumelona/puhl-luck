/// HDC (Hyperdimensional Computing) operations — pyo3 0.23 compatible

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use ndarray::Array1;

const HDC_WORD_BITS: usize = 64;

#[pyfunction]
pub fn feature_hv_rust(py: Python, feature: &str, words: usize) -> PyResult<Py<PyArray1<u64>>> {
    let hv = generate_feature_hv(feature, words);
    Ok(Array1::from_vec(hv).into_pyarray(py).into())
}

#[inline(always)]
pub fn generate_feature_hv(feature: &str, words: usize) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    feature.hash(&mut hasher);
    let seed = hasher.finish();
    let mut result = Vec::with_capacity(words);
    for i in 0..words {
        let mut h = DefaultHasher::new();
        seed.hash(&mut h);
        i.hash(&mut h);
        result.push(h.finish());
    }
    result
}

#[pyfunction]
pub fn rotate_hv_rust<'py>(
    py: Python<'py>,
    value: PyReadonlyArray1<u64>,
    amount: i32,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let arr = value.as_array().to_owned();
    let rotated = rotate_hv_internal(arr, amount);
    Ok(rotated.into_pyarray(py))
}

#[inline(always)]
pub fn rotate_hv_internal(value: Array1<u64>, amount: i32) -> Array1<u64> {
    if value.is_empty() { return value; }
    let bit_shift = (amount.rem_euclid(HDC_WORD_BITS as i32)) as u32;
    if bit_shift == 0 { return value; }
    value.mapv(|x| (x << bit_shift) | (x >> (HDC_WORD_BITS as u32 - bit_shift)))
}

#[pyfunction]
pub fn bundle_hv_rust(py: Python, features: Vec<String>, words: usize) -> PyResult<Py<PyArray1<u64>>> {
    let mut result = vec![0u64; words];
    for (i, feature) in features.iter().enumerate() {
        let fhv = generate_feature_hv(feature, words);
        let rotated = rotate_hv_internal(Array1::from_vec(fhv), i as i32);
        for (j, &v) in rotated.iter().enumerate() {
            result[j] ^= v;
        }
    }
    Ok(Array1::from_vec(result).into_pyarray(py).into())
}

#[pyfunction]
pub fn hv_similarity_rust(
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<f64> {
    let a = a.as_array();
    let b = b.as_array();
    if a.is_empty() || b.is_empty() { return Ok(0.0); }
    let words = a.len().min(b.len());
    let bits = words * HDC_WORD_BITS;
    let diff: u32 = a.iter().zip(b.iter()).take(words).map(|(x, y)| (x ^ y).count_ones()).sum();
    Ok(1.0 - (diff as f64 / bits as f64))
}
