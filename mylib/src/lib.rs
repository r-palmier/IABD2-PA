use pyo3::prelude::*;

#[pyfunction]
fn sum_vector(v: Vec<f32>) -> f32 {
    v.iter().sum()
}

#[pymodule]
fn mylib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_vector, m)?)?;
    Ok(())
}
