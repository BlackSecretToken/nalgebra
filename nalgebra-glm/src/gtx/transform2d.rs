use na::UnitComplex;

use crate::aliases::{TMat3, TVec2};
use crate::traits::{Number, RealNumber};

/// Builds a 2D rotation matrix from an angle and right-multiply it to `m`.
///
/// # See also:
///
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scale2d`](fn.scale2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translate2d`](fn.translate2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn rotate2d<T: RealNumber>(m: &TMat3<T>, angle: T) -> TMat3<T> {
    m * UnitComplex::new(angle).to_homogeneous()
}

/// Builds a 2D scaling matrix and right-multiply it to `m`.
///
/// # See also:
///
/// * [`rotate2d`](fn.rotate2d.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translate2d`](fn.translate2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn scale2d<T: Number>(m: &TMat3<T>, v: &TVec2<T>) -> TMat3<T> {
    m.prepend_nonuniform_scaling(v)
}

/// Builds a translation matrix and right-multiply it to `m`.
///
/// # See also:
///
/// * [`rotate2d`](fn.rotate2d.html)
/// * [`rotation2d`](fn.rotation2d.html)
/// * [`scale2d`](fn.scale2d.html)
/// * [`scaling2d`](fn.scaling2d.html)
/// * [`translation2d`](fn.translation2d.html)
pub fn translate2d<T: Number>(m: &TMat3<T>, v: &TVec2<T>) -> TMat3<T> {
    m.prepend_translation(v)
}
