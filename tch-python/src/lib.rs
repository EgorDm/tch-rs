use std::ffi::c_void;
use pyo3::prelude::*;
use pyo3::{AsPyPointer, ffi, wrap_pyfunction};
use pyo3::exceptions::PyTypeError;
use torch_sys::{thp_variable_check, thp_variable_unpack, thp_variable_wrap};
use tch::{Tensor};

#[pyfunction]
fn print(py: Python, x: PyObject) -> PyResult<PyObject> {
    if unsafe { thp_variable_check(x.as_ptr() as *mut c_void) } {
        let unwrapped = unsafe { thp_variable_unpack(x.as_ptr() as *mut c_void) };
        let mut tensor = unsafe { Tensor::from_ptr(unwrapped) };

        println!("Valid tensor");
        tensor.print();
        tensor.grad().print();

        let result = unsafe { thp_variable_wrap(tensor.as_mut_ptr()) };

        unsafe {
            Ok(PyObject::from_owned_ptr(py, result as *mut ffi::PyObject))
        }
    } else {
        println!("Not a valid tensor!!");
        Err(PyTypeError::new_err("Not a valid tensor"))
    }

    // https://github.com/pytorch/pytorch/blob/40d1f77384672337bd7e734e32cb5fad298959bd/torch/csrc/autograd/python_variable.cpp#L239
    // C++ owns the object
    // https://github.com/pytorch/pytorch/blob/717d8c6224762cf1d745fa2a0910b29e2a4b10f3/torch/csrc/autograd/python_variable.cpp#L192
    // TensorBase destroyOwned

}

#[pymodule]
fn tchpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(print))?;
    Ok(())
}