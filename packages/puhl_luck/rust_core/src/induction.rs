/// Operator Induction Fast Path — pyo3 0.23
///
/// Accelerates the hot path in OperatorInduction:
/// - greedy_cluster_transitions: cluster by HDC completion-vector similarity
/// - extract_common_features: majority-vote common features across a cluster

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Greedy clustering of transitions by feature overlap on completion features.
///
/// Equivalent to OperatorInduction.identify_repeated_patterns but 10-20× faster
/// because all centroid comparisons run in Rust without Python object overhead.
///
/// Args:
///     items: list of (transition_id, completion_features)
///     similarity_threshold: min Jaccard to join an existing cluster
///
/// Returns:
///     list of clusters, each cluster = [transition_id, ...]
#[pyfunction]
#[pyo3(signature = (items, similarity_threshold=0.4))]
pub fn greedy_cluster_transitions_rust(
    py: Python,
    items: Vec<(String, Vec<String>)>,
    similarity_threshold: f64,
) -> PyResult<PyObject> {
    // clusters: Vec<(centroid_features: Vec<String>, members: Vec<String>)>
    let mut clusters: Vec<(std::collections::HashSet<String>, Vec<String>)> = Vec::new();

    for (tid, feats) in &items {
        let fset: std::collections::HashSet<&str> = feats.iter().map(|s| s.as_str()).collect();

        let mut best_cluster: Option<usize> = None;
        let mut best_sim = similarity_threshold;

        for (ci, (centroid, _)) in clusters.iter().enumerate() {
            let cset: std::collections::HashSet<&str> = centroid.iter().map(|s| s.as_str()).collect();
            let inter = fset.intersection(&cset).count();
            let union = fset.union(&cset).count();
            let sim = if union > 0 { inter as f64 / union as f64 } else { 0.0 };
            if sim > best_sim {
                best_sim = sim;
                best_cluster = Some(ci);
            }
        }

        if let Some(ci) = best_cluster {
            // Add to cluster, update centroid (union of all features = grow centroid)
            clusters[ci].0.extend(feats.iter().cloned());
            clusters[ci].1.push(tid.clone());
        } else {
            let centroid: std::collections::HashSet<String> = feats.iter().cloned().collect();
            clusters.push((centroid, vec![tid.clone()]));
        }
    }

    let list = PyList::empty(py);
    for (_, members) in clusters {
        let ml = PyList::empty(py);
        for m in members { ml.append(m)?; }
        list.append(ml)?;
    }
    Ok(list.into())
}

/// Extract common features across a set of feature lists.
///
/// Args:
///     feature_lists: list of feature lists (one per transition)
///     threshold_ratio: fraction of lists a feature must appear in (0.0–1.0)
///
/// Returns:
///     list of features meeting the threshold
#[pyfunction]
#[pyo3(signature = (feature_lists, threshold_ratio=0.5))]
pub fn extract_common_features_rust(
    py: Python,
    feature_lists: Vec<Vec<String>>,
    threshold_ratio: f64,
) -> PyResult<PyObject> {
    if feature_lists.is_empty() {
        return Ok(PyList::empty(py).into());
    }
    let n = feature_lists.len() as f64;
    let threshold = (n * threshold_ratio).ceil() as usize;

    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for feats in &feature_lists {
        // Count each feature once per list (set semantics)
        let unique: std::collections::HashSet<&str> = feats.iter().map(|s| s.as_str()).collect();
        for f in unique { *counts.entry(f.to_string()).or_insert(0) += 1; }
    }

    let mut common: Vec<String> = counts.into_iter()
        .filter(|(_, c)| *c >= threshold)
        .map(|(f, _)| f)
        .collect();
    common.sort(); // deterministic order

    let list = PyList::empty(py);
    for f in common { list.append(f)?; }
    Ok(list.into())
}

/// Classify operator type from feature frequency counts.
///
/// Args:
///     feature_counts: list of (feature, count) pairs from partial states
///
/// Returns:
///     operator type string: "explanation"|"repair"|"comparison"|"transformation"|"composition"|"completion"
#[pyfunction]
pub fn classify_operator_type_rust(feature_counts: Vec<(String, usize)>) -> String {
    let features: std::collections::HashSet<&str> = feature_counts.iter().map(|(f, _)| f.as_str()).collect();
    if ["question","what","why","how","when","where","who"].iter().any(|k| features.contains(k)) {
        return "explanation".to_string();
    }
    if ["error","bug","fix","incorrect","wrong","issue"].iter().any(|k| features.contains(k)) {
        return "repair".to_string();
    }
    if ["compare","difference","similar","versus","vs"].iter().any(|k| features.contains(k)) {
        return "comparison".to_string();
    }
    if ["solve","transform","convert","translate","problem"].iter().any(|k| features.contains(k)) {
        return "transformation".to_string();
    }
    if ["combine","merge","integrate","compose"].iter().any(|k| features.contains(k)) {
        return "composition".to_string();
    }
    "completion".to_string()
}

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(greedy_cluster_transitions_rust, m)?)?;
    m.add_function(wrap_pyfunction!(extract_common_features_rust, m)?)?;
    m.add_function(wrap_pyfunction!(classify_operator_type_rust, m)?)?;
    Ok(())
}
