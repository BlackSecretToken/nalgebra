extern crate rand;
extern crate nalgebra as na;

extern crate serde_json;

use na::{
    DMatrix,
    Matrix3x4,
    Point3,
    Translation3,
    Rotation3,
    Isometry3,
    IsometryMatrix3,
    Similarity3,
    SimilarityMatrix3,
    Quaternion
};

macro_rules! test_serde(
    ($($test: ident, $ty: ident);* $(;)*) => {$(
        #[test]
        fn $test() {
            let v: $ty<f32> = rand::random();
            let serialized = serde_json::to_string(&v).unwrap();
            assert_eq!(v, serde_json::from_str(&serialized).unwrap());
        }
    )*}
);

#[test]
fn serde_dmatrix() {
    let v: DMatrix<f32> = DMatrix::new_random(3, 4);
    let serialized = serde_json::to_string(&v).unwrap();
    assert_eq!(v, serde_json::from_str(&serialized).unwrap());
}

test_serde!(
    serde_matrix3x4,          Matrix3x4;
    serde_point3,             Point3;
    serde_translation3,       Translation3;
    serde_rotation3,          Rotation3;
    serde_isometry3,          Isometry3;
    serde_isometry_matrix3,   IsometryMatrix3;
    serde_similarity3,        Similarity3;
    serde_similarity_matrix3, SimilarityMatrix3;
    serde_quaternion,         Quaternion;
);
