#![allow(dead_code)]
#![allow(unstable_features)]

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
    pub fn new(values: Vec<T>, shape: Vec<usize>) -> Self {
        let ndim = shape.len();

        Self {
            storage: Rc::new(TensorStorage::from(values)),
            shape,
            ndim,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let num_elements = shape.iter().fold(1, |total, val| total * val);
        let storage = TensorStorage::zeros(num_elements);

        Self {
            storage: Rc::new(storage),
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let num_elements = shape.iter().fold(1, |total, val| total * val);
        let storage = TensorStorage::ones(num_elements);

        Self {
            storage: Rc::new(storage),
            shape: shape.clone(),
            ndim: shape.len(),
        }
    }

    // todo
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

    // todo
    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

fn get_sequential_tensor<T>(shape: Vec<usize>) -> Tensor<T>
where
    T: TensorValue,
{
    let size = shape.iter().fold(1, |a, b| a * b);

    let mut values = Vec::<T>::new();
    let mut current_value = T::zero();

    for _ in 0..size {
        values.push(current_value.clone());
        current_value = current_value + T::one();
    }

    Tensor::new(values, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_new() {
        // todo: useless test?
        let tensor = Tensor::<i64>::new(vec![1, 2, 3, 4], vec![2, 2]);
        assert_eq!(tensor.ndim, 2);
    }

    #[test]
    fn create_zeros() {
        let tensor = Tensor::<i64>::zeros(vec![4, 4, 3]);
        assert_eq!(tensor.ndim, 3);
        assert_eq!(tensor.storage.len(), 4 * 4 * 3);
        assert_eq!(tensor.shape, vec![4, 4, 3]);
    }

    #[test]
    fn iterator_sequential() {
        let tensor: Tensor<i64> = get_sequential_tensor(vec![3, 3, 3]);
        assert_eq!(TensorIter::new(&tensor).count(), 3 * 3 * 3);

        for i in TensorIter::new(&tensor) {
            assert_eq!(i, 0);
        }
    }

    #[test]
    fn iterator_strided() {
        todo!("not implemented yet. stride operation not usable yet.");
    }

    #[test]
    fn get_row() {
        let original_tensor = Tensor::<i64>::zeros(vec![2, 3, 3]);

        let tensor = original_tensor.row(1);
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.shape, vec![3, 3]);
        todo!("missing value checks");
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
