/// PUHL-LUCK Rust Core
///
/// High-performance implementations of:
/// - HDC (Hyperdimensional Computing) operations
/// - Field energy computation
/// - Resonance calculation
/// - Field operations and dynamics
/// - Pattern matching and search
/// - State completion
/// - Memory management and pruning
/// - Transition learning (new: expose_pair fast path)
/// - Operator induction (new: clustering fast path)

use pyo3::prelude::*;

mod hdc;
mod energy;
mod resonance;
mod field;
mod matching;
mod completion;
mod memory_management;
mod transition;
mod induction;

/// Python module initialization
#[pymodule]
fn puhl_luck_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // HDC operations
    m.add_function(wrap_pyfunction!(hdc::feature_hv_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hdc::rotate_hv_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hdc::bundle_hv_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hdc::hv_similarity_rust, m)?)?;

    // Energy computation
    energy::register_functions(m)?;

    // Resonance calculation
    resonance::register_functions(m)?;

    // Field operations
    field::register_functions(m)?;

    // Pattern matching
    matching::register_functions(m)?;

    // State completion
    completion::register_functions(m)?;

    // Memory management
    memory_management::register_functions(m)?;

    // Transition memory (new)
    transition::register_functions(m)?;

    // Operator induction (new)
    induction::register_functions(m)?;

    Ok(())
}
