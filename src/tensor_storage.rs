#![allow(dead_code)]

use num_traits::Num;

// Notes: This looks weird to me but clears up all traits required
pub trait TensorValue: Clone + Default + Num {}

impl<T: Clone + Default + Num> TensorValue for T {}

pub struct TensorStorage<T> {
    storage: Vec<T>,
}

impl<T> TensorStorage<T>
where
    T: TensorValue,
{
    pub fn new() -> Self {
        Self { storage: vec![] }
    }

    pub fn ones(size: usize) -> Self {
        let v = vec![T::one(); size];
        Self::from(v)
    }

    pub fn zeros(size: usize) -> Self {
        let v = vec![T::zero(); size];
        Self::from(v)
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }
}

impl<T> From<Vec<T>> for TensorStorage<T> {
    fn from(values: Vec<T>) -> Self {
        Self { storage: values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_construct() {
        let storage = TensorStorage::<i64>::new();
        assert_eq!(storage.len(), 0);
    }

    #[test]
    fn storage_construct_from() {
        let storage = TensorStorage::<i64>::from(vec![1; 128]);
        assert_eq!(storage.len(), 128);
        assert_eq!(storage.storage, vec![1; 128]);
    }
}
