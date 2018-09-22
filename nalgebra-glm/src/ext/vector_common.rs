use na::{self, DefaultAllocator};

use traits::{Alloc, Number, Dimension};
use aliases::Vec;

/// Component-wise maximum between a vector and a scalar.
pub fn max<N: Number, D: Dimension>(a: &Vec<N, D>, b: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    a.map(|a| na::sup(&a, &b))
}

/// Component-wise maximum between two vectors.
pub fn max2<N: Number, D: Dimension>(a: &Vec<N, D>, b: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    na::sup(a, b)
}

/// Component-wise maximum between three vectors.
pub fn max3<N: Number, D: Dimension>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    max2(&max2(a, b), c)
}

/// Component-wise maximum between four vectors.
pub fn max4<N: Number, D: Dimension>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>, d: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    max2(&max2(a, b), &max2(c, d))
}

/// Component-wise minimum between a vector and a scalar.
pub fn min<N: Number, D: Dimension>(x: &Vec<N, D>, y: N) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    x.map(|x| na::inf(&x, &y))
}

/// Component-wise minimum between two vectors.
pub fn min2<N: Number, D: Dimension>(x: &Vec<N, D>, y: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    na::inf(x, y)
}

/// Component-wise minimum between three vectors.
pub fn min3<N: Number, D: Dimension>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    min2(&min2(a, b), c)
}

/// Component-wise minimum between four vectors.
pub fn min4<N: Number, D: Dimension>(a: &Vec<N, D>, b: &Vec<N, D>, c: &Vec<N, D>, d: &Vec<N, D>) -> Vec<N, D>
    where DefaultAllocator: Alloc<N, D> {
    min2(&min2(a, b), &min2(c, d))
}
