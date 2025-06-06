use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn sum_floats(values: Vec<f64>) -> f64 {
    values.iter().sum()
}

#[pymodule]
fn pyo3_sum(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_floats, m)?)?;
    Ok(())
}
