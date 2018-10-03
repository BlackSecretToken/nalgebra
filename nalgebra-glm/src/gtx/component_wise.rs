use na::{self, DefaultAllocator};

use traits::{Number, Alloc, Dimension};
use aliases::TMat;

/// The sum of every component of the given matrix or vector.
///
/// # See also:
///
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`comp_mul`](fn.comp_mul.html)
pub fn comp_add<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::zero(), |x, y| x + *y)
}

/// The maximum of every component of the given matrix or vector.
///
/// # See also:
///
/// * [`comp_add`](fn.comp_add.html)
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
/// * [`max`](fn.max.html)
/// * [`max2`](fn.max2.html)
/// * [`max3`](fn.max3.html)
/// * [`max4`](fn.max4.html)
pub fn comp_max<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::min_value(), |x, y| na::sup(&x, y))
}

/// The minimum of every component of the given matrix or vector.
///
/// # See also:
///
/// * [`comp_add`](fn.comp_add.html)
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_mul`](fn.comp_mul.html)
/// * [`min`](fn.min.html)
/// * [`min2`](fn.min2.html)
/// * [`min3`](fn.min3.html)
/// * [`min4`](fn.min4.html)
pub fn comp_min<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::max_value(), |x, y| na::inf(&x, y))
}

/// The product of every component of the given matrix or vector.
///
/// # See also:
///
/// * [`comp_add`](fn.comp_add.html)
/// * [`comp_max`](fn.comp_max.html)
/// * [`comp_min`](fn.comp_min.html)
pub fn comp_mul<N: Number, R: Dimension, C: Dimension>(m: &TMat<N, R, C>) -> N
    where DefaultAllocator: Alloc<N, R, C> {
    m.iter().fold(N::one(), |x, y| x * *y)
}

//pub fn vec< L, floatType, Q > compNormalize (vec< L, T, Q > const &v)
//pub fn vec< L, T, Q > compScale (vec< L, floatType, Q > const &v)
