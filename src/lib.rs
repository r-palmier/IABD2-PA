use pyo3::prelude::*;

#[pyfunction]
fn test(values: Vec<f32>) -> f32 {
    values.into_iter().sum()
}

#[pymodule]
fn vecsum(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test, m)?)?;
    Ok(())
}
