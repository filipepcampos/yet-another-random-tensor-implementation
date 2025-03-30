#![allow(dead_code)]

use std::ops::Add;
use std::rc::Rc;

pub mod tensor_storage;
use crate::tensor_storage::TensorStorage;
use crate::tensor_storage::TensorValue;

struct Tensor<T> {
    storage: Rc<TensorStorage<T>>,
    shape: Vec<usize>,
    ndim: usize,
}

impl<T> Tensor<T>
where
    T: TensorValue,
{
    pub fn zeros(shape: Vec<usize>) -> Tensor<T> {
        let num_elements = shape.iter().fold(1, |total, val| total * val);
        let storage = TensorStorage::zeros(num_elements);
        Tensor {
            storage: Rc::new(storage),
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn ones(shape: Vec<usize>) -> Tensor<T> {
        let num_elements = shape.iter().fold(1, |total, val| total * val);
        let storage = TensorStorage::ones(num_elements);
        Tensor {
            storage: Rc::new(storage),
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn row(&self, n: usize) -> Self {
        assert!(n < self.ndim);

        Self {
            storage: self.storage.clone(),
            shape: self.shape[1..].to_vec(),
            ndim: self.ndim - 1,
        }
    }
}

impl<T> Add for &Tensor<T>
where
    T: TensorValue,
{
    type Output = Tensor<T>;

    // todo: rhs from different type
    // todo: inplace modification
    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let v: Vec<T> = Iterator::zip(TensorIter::new(&self), TensorIter::new(&rhs))
            .map(|(l, r)| l + r)
            .collect();

        let storage = Rc::new(TensorStorage::from(v));

        Self::Output {
            storage,
            shape: self.shape.clone(),
            ndim: self.ndim,
        }
    }
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(), // TODO clone contents, not Rc
            shape: self.shape.clone(),
            ndim: self.ndim.clone(),
        }
    }
}

struct TensorIter<'a, T> {
    tensor: &'a Tensor<T>,
    current: usize,
}

impl<'a, T> TensorIter<'a, T> {
    fn new(tensor: &'a Tensor<T>) -> Self {
        Self { tensor, current: 0 }
    }
}

impl<'a, T> Iterator for TensorIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_zeros() {
        let tensor = Tensor::<i64>::zeros(vec![4, 4, 3]);
        assert_eq!(tensor.ndim, 3);
        assert_eq!(tensor.storage.len(), 4 * 4 * 3);
        assert_eq!(tensor.shape, vec![4, 4, 3]);
    }

    #[test]
    fn iterate() {
        let tensor = Tensor::<i64>::zeros(vec![3, 3, 3]);
        for i in TensorIter::new(&tensor) {
            assert_eq!(i, 0);
        }
    }

    #[test]
    fn get_row() {
        let original_tensor = Tensor::<i64>::zeros(vec![2, 3, 3]);
        let tensor = original_tensor.row(1);
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.shape, vec![3, 3]);
    }

    #[test]
    fn add_same() {
        let t1 = Tensor::<i64>::ones(vec![3, 3, 3]);
        let t2 = Tensor::<i64>::ones(vec![3, 3, 3]);

        let result = &t1 + &t2;

        assert_eq!(result.shape, t1.shape);
        assert_eq!(result.shape, t2.shape);
        assert_eq!(result.ndim, t1.ndim);
        for i in TensorIter::new(&result) {
            assert_eq!(i, 2);
        }
    }
}
