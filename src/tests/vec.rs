use std::num::{Zero, One};
use std::rand::{random};
use std::cmp::ApproxEq;
use na::*;

macro_rules! test_iterator_impl(
    ($t: ty, $n: ty) => (
        do 10000.times {
            let v: $t      = random();
            let mut mv: $t = v.clone();
            let n: $n      = random();

            let nv: $t = v.iter().map(|e| e * n).collect();

            for e in mv.mut_iter() {
                *e = *e * n
            }

            assert!(nv == mv && nv == v * n);
        }
    )
)

macro_rules! test_commut_dot_impl(
    ($t: ty) => (
        do 10000.times {
            let v1 : $t = random();
            let v2 : $t = random();
        
            assert!(v1.dot(&v2).approx_eq(&v2.dot(&v1)));
        }
    );
)

macro_rules! test_scalar_op_impl(
    ($t: ty, $n: ty) => (
        do 10000.times {
            let v1 : $t = random();
            let n  : $n = random();
        
            assert!(((v1 * n) / n).approx_eq(&v1));
            assert!(((v1 / n) * n).approx_eq(&v1));
            assert!(((v1 - n) + n).approx_eq(&v1));
            assert!(((v1 + n) - n).approx_eq(&v1));

            let mut v1 : $t = random();
            let v0 : $t = v1.clone();
            let n  : $n = random();

            v1 = v1 * n;
            v1 = v1 / n;
        
            assert!(v1.approx_eq(&v0));
        }
    );
)

macro_rules! test_basis_impl(
    ($t: ty) => (
        do 10000.times {
            do Basis::canonical_basis |e1: $t| {
                do Basis::canonical_basis |e2: $t| {
                    assert!(e1 == e2 || e1.dot(&e2).approx_eq(&Zero::zero()));

                    true
                }

                assert!(e1.norm().approx_eq(&One::one()));

                true
            }
        }
    );
)

macro_rules! test_subspace_basis_impl(
    ($t: ty) => (
        do 10000.times {
            let v : $t = random();
            let v1     = v.normalized();

            do v1.orthonormal_subspace_basis() |e1| {
                // check vectors are orthogonal to v1
                assert!(v1.dot(&e1).approx_eq(&Zero::zero()));
                // check vectors form an orthonormal basis
                assert!(e1.norm().approx_eq(&One::one()));
                // check vectors form an ortogonal basis
                do v1.orthonormal_subspace_basis() |e2| {
                    assert!(e1 == e2 || e1.dot(&e2).approx_eq(&Zero::zero()));

                    true
                }

                true
            }
        }
    );
)

#[test]
fn test_cross_vec3() {
    do 10000.times {
        let v1 : Vec3<f64> = random();
        let v2 : Vec3<f64> = random();
        let v3 : Vec3<f64> = v1.cross(&v2);

        assert!(v3.dot(&v2).approx_eq(&Zero::zero()));
        assert!(v3.dot(&v1).approx_eq(&Zero::zero()));
    }
}

#[test]
fn test_commut_dot_vec0() {
    test_commut_dot_impl!(Vec0<f64>);
}

#[test]
fn test_commut_dot_vec1() {
    test_commut_dot_impl!(Vec1<f64>);
}

#[test]
fn test_commut_dot_vec2() {
    test_commut_dot_impl!(Vec2<f64>);
}

#[test]
fn test_commut_dot_vec3() {
    test_commut_dot_impl!(Vec3<f64>);
}

#[test]
fn test_commut_dot_vec4() {
    test_commut_dot_impl!(Vec4<f64>);
}

#[test]
fn test_commut_dot_vec5() {
    test_commut_dot_impl!(Vec5<f64>);
}

#[test]
fn test_commut_dot_vec6() {
    test_commut_dot_impl!(Vec6<f64>);
}

#[test]
fn test_basis_vec0() {
    test_basis_impl!(Vec0<f64>);
}

#[test]
fn test_basis_vec1() {
    test_basis_impl!(Vec1<f64>);
}

#[test]
fn test_basis_vec2() {
    test_basis_impl!(Vec2<f64>);
}

#[test]
fn test_basis_vec3() {
    test_basis_impl!(Vec3<f64>);
}

#[test]
fn test_basis_vec4() {
    test_basis_impl!(Vec4<f64>);
}

#[test]
fn test_basis_vec5() {
    test_basis_impl!(Vec5<f64>);
}

#[test]
fn test_basis_vec6() {
    test_basis_impl!(Vec6<f64>);
}

#[test]
fn test_subspace_basis_vec0() {
    test_subspace_basis_impl!(Vec0<f64>);
}

#[test]
fn test_subspace_basis_vec1() {
    test_subspace_basis_impl!(Vec1<f64>);
}

#[test]
fn test_subspace_basis_vec2() {
    test_subspace_basis_impl!(Vec2<f64>);
}

#[test]
fn test_subspace_basis_vec3() {
    test_subspace_basis_impl!(Vec3<f64>);
}

#[test]
fn test_subspace_basis_vec4() {
    test_subspace_basis_impl!(Vec4<f64>);
}

#[test]
fn test_subspace_basis_vec5() {
    test_subspace_basis_impl!(Vec5<f64>);
}

#[test]
fn test_subspace_basis_vec6() {
    test_subspace_basis_impl!(Vec6<f64>);
}

#[test]
fn test_scalar_op_vec0() {
    test_scalar_op_impl!(Vec0<f64>, f64);
}

#[test]
fn test_scalar_op_vec1() {
    test_scalar_op_impl!(Vec1<f64>, f64);
}

#[test]
fn test_scalar_op_vec2() {
    test_scalar_op_impl!(Vec2<f64>, f64);
}
 
#[test]
fn test_scalar_op_vec3() {
    test_scalar_op_impl!(Vec3<f64>, f64);
}
 
#[test]
fn test_scalar_op_vec4() {
    test_scalar_op_impl!(Vec4<f64>, f64);
}
 
#[test]
fn test_scalar_op_vec5() {
    test_scalar_op_impl!(Vec5<f64>, f64);
}
 
#[test]
fn test_scalar_op_vec6() {
    test_scalar_op_impl!(Vec6<f64>, f64);
}

#[test]
fn test_iterator_vec0() {
    test_iterator_impl!(Vec0<f64>, f64);
}

#[test]
fn test_iterator_vec1() {
    test_iterator_impl!(Vec1<f64>, f64);
}

#[test]
fn test_iterator_vec2() {
    test_iterator_impl!(Vec2<f64>, f64);
}
 
#[test]
fn test_iterator_vec3() {
    test_iterator_impl!(Vec3<f64>, f64);
}
 
#[test]
fn test_iterator_vec4() {
    test_iterator_impl!(Vec4<f64>, f64);
}
 
#[test]
fn test_iterator_vec5() {
    test_iterator_impl!(Vec5<f64>, f64);
}
 
#[test]
fn test_iterator_vec6() {
    test_iterator_impl!(Vec6<f64>, f64);
}

#[test]
fn test_ord_vec3() {
    // equality
    assert!(Vec3::new(0.5, 0.5, 0.5) == Vec3::new(0.5, 0.5, 0.5));
    assert!(!(Vec3::new(1.5, 0.5, 0.5) == Vec3::new(0.5, 0.5, 0.5)));
    assert!(Vec3::new(1.5, 0.5, 0.5) != Vec3::new(0.5, 0.5, 0.5));

    // comparable
    assert!(Vec3::new(0.5, 0.3, 0.3) < Vec3::new(1.0, 2.0, 1.0));
    assert!(Vec3::new(0.5, 0.3, 0.3) <= Vec3::new(1.0, 2.0, 1.0));
    assert!(Vec3::new(2.0, 4.0, 2.0) > Vec3::new(1.0, 2.0, 1.0));
    assert!(Vec3::new(2.0, 4.0, 2.0) >= Vec3::new(1.0, 2.0, 1.0));

    // not comparable
    assert!(!(Vec3::new(0.0, 3.0, 0.0) < Vec3::new(1.0, 2.0, 1.0)));
    assert!(!(Vec3::new(0.0, 3.0, 0.0) > Vec3::new(1.0, 2.0, 1.0)));
    assert!(!(Vec3::new(0.0, 3.0, 0.0) <= Vec3::new(1.0, 2.0, 1.0)));
    assert!(!(Vec3::new(0.0, 3.0, 0.0) >= Vec3::new(1.0, 2.0, 1.0)));
}

#[test]
fn test_min_max_vec3() {
    assert_eq!(Vec3::new(1, 2, 3).max(&Vec3::new(3, 2, 1)), Vec3::new(3, 2, 3));
    assert_eq!(Vec3::new(1, 2, 3).min(&Vec3::new(3, 2, 1)), Vec3::new(1, 2, 1));
    assert_eq!(
        Vec3::new(0, 2, 4).clamp(
            &Vec3::new(1, 1, 1), &Vec3::new(3, 3, 3)
        ), Vec3::new(1, 2, 3)
    );
}

#[test]
fn test_outer_vec3() {
    assert_eq!(
        Vec3::new(1, 2, 3).outer(&Vec3::new(4, 5, 6)),
        Mat3::new(
            4, 5, 6,
            8, 10, 12,
            12, 15, 18));
}
