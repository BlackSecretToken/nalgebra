use num::{One, Zero};
use std::ops::{Div, DivAssign, Index, IndexMut, Mul, MulAssign};

use simba::scalar::{ClosedAdd, ClosedMul, RealField, SubsetOf};

use crate::base::allocator::Allocator;
use crate::base::dimension::{DimNameAdd, DimNameSum, U1, U4};
use crate::base::{CVectorN, Const, DefaultAllocator, MatrixN, Scalar};

use crate::geometry::{
    Isometry, Point, Rotation, Similarity, SubTCategoryOf, SuperTCategoryOf, TAffine, TCategory,
    TCategoryMul, TGeneral, TProjective, Transform, Translation, UnitQuaternion,
};

/*
 *
 * In the following, we provide:
 * =========================
 *
 * Index<(usize, usize)>
 * IndexMut<(usize, usize)> (where TCategory == TGeneral)
 *
 * (Operators)
 *
 * Transform × Isometry
 * Transform × Rotation
 * Transform × Similarity
 * Transform × Transform
 * Transform × UnitQuaternion
 * TODO: Transform × UnitComplex
 * Transform × Translation
 * Transform × Vector
 * Transform × Point
 *
 * Isometry       × Transform
 * Rotation       × Transform
 * Similarity     × Transform
 * Translation    × Transform
 * UnitQuaternion × Transform
 * TODO: UnitComplex × Transform
 *
 * TODO: Transform ÷ Isometry
 * Transform ÷ Rotation
 * TODO: Transform ÷ Similarity
 * Transform ÷ Transform
 * Transform ÷ UnitQuaternion
 * Transform ÷ Translation
 *
 * TODO: Isometry       ÷ Transform
 * Rotation       ÷ Transform
 * TODO: Similarity     ÷ Transform
 * Translation    ÷ Transform
 * UnitQuaternion ÷ Transform
 * TODO: UnitComplex ÷ Transform
 *
 *
 * (Assignment Operators)
 *
 *
 * Transform ×= Transform
 * Transform ×= Similarity
 * Transform ×= Isometry
 * Transform ×= Rotation
 * Transform ×= UnitQuaternion
 * TODO: Transform ×= UnitComplex
 * Transform ×= Translation
 *
 * Transform ÷= Transform
 * TODO: Transform ÷= Similarity
 * TODO: Transform ÷= Isometry
 * Transform ÷= Rotation
 * Transform ÷= UnitQuaternion
 * TODO: Transform ÷= UnitComplex
 *
 */

/*
 *
 * Indexing.
 *
 */
impl<N: RealField, C: TCategory, const D: usize> Index<(usize, usize)> for Transform<N, C, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    type Output = N;

    #[inline]
    fn index(&self, ij: (usize, usize)) -> &N {
        self.matrix().index(ij)
    }
}

// Only general transformations are mutably indexable.
impl<N: RealField, const D: usize> IndexMut<(usize, usize)> for Transform<N, TGeneral, D>
where
    Const<D>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    #[inline]
    fn index_mut(&mut self, ij: (usize, usize)) -> &mut N {
        self.matrix_mut().index_mut(ij)
    }
}

// Transform × Vector
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, C, D>, rhs: CVectorN<N, D>, Output = CVectorN<N, D>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let transform = self.matrix().fixed_slice::<Const<D>, Const<D>>(0, 0);

        if C::has_normalizer() {
            let normalizer = self.matrix().fixed_slice::<U1, Const<D>>(D, 0);
            let n = normalizer.tr_dot(&rhs);

            if !n.is_zero() {
                return transform * (rhs / n);
            }
        }

        transform * rhs
    };
);

// Transform × Point
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, C, D>, rhs: Point<N, D>, Output = Point<N, D>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let transform   = self.matrix().fixed_slice::<Const<D>, Const<D>>(0, 0);
        let translation = self.matrix().fixed_slice::<Const<D>, U1>(0, D);

        if C::has_normalizer() {
            let normalizer = self.matrix().fixed_slice::<U1, Const<D>>(D, 0);
            #[allow(clippy::suspicious_arithmetic_impl)]
            let n = normalizer.tr_dot(&rhs.coords) + unsafe { *self.matrix().get_unchecked((D, D)) };

            if !n.is_zero() {
                return (transform * rhs + translation) / n;
            }
        }

        transform * rhs + translation
    };
);

// Transform × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for CA, CB;
    where Const<D>: DimNameAdd<U1>, CA: TCategoryMul<CB>, CB: TCategory;
    self: Transform<N, CA, D>, rhs: Transform<N, CB, D>, Output = Transform<N, CA::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.matrix());
);

// Transform × Rotation
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, Const<D>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, C, D>, rhs: Rotation<N, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Rotation × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (Const<D>, Const<D>), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Rotation<N, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform × UnitQuaternion
md_impl_all!(
    Mul, mul where N: RealField;
    (U4, U4), (U4, U1)
    const;
    for C;
    where C: TCategoryMul<TAffine>;
    self: Transform<N, C, 3>, rhs: UnitQuaternion<N>, Output = Transform<N, C::Representative, 3>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// UnitQuaternion × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (U4, U1), (U4, U4)
    const;
    for C;
    where C: TCategoryMul<TAffine>;
    self: UnitQuaternion<N>, rhs: Transform<N, C, 3>, Output = Transform<N, C::Representative, 3>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform × Isometry
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C, R;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
    self: Transform<N, C, D>, rhs: Isometry<N, R, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Isometry × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (Const<D>, U1), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for C, R;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
    self: Isometry<N, R, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform × Similarity
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C, R;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
    self: Transform<N, C, D>, rhs: Similarity<N, R, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Similarity × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (Const<D>, U1), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for C, R;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
    self: Similarity<N, R, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

/*
 *
 * TODO: don't explicitly build the homogeneous translation matrix.
 * Directly apply the translation, just as in `Matrix::{append,prepend}_translation`. This has not
 * been done yet because of the `DimNameDiff` requirement (which is not automatically deduced from
 * `DimNameAdd` requirement).
 *
 */
// Transform × Translation
md_impl_all!(
    Mul, mul where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, C, D>, rhs: Translation<N, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
    [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
    [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
);

// Translation × Transform
md_impl_all!(
    Mul, mul where N: RealField;
    (Const<D>, U1), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Translation<N, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
    [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
    [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
    [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
);

// Transform ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for CA, CB;
    where Const<D>: DimNameAdd<U1>, CA: TCategoryMul<CB>, CB: SubTCategoryOf<TProjective>;
    self: Transform<N, CA, D>, rhs: Transform<N, CB, D>, Output = Transform<N, CA::Representative, D>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.clone().inverse() };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.clone().inverse() };
);

// Transform ÷ Rotation
md_impl_all!(
    Div, div where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, Const<D>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, C, D>, rhs: Rotation<N, D>, Output = Transform<N, C::Representative, D>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
);

// Rotation ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (Const<D>, Const<D>), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Rotation<N, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
);

// Transform ÷ UnitQuaternion
md_impl_all!(
    Div, div where N: RealField;
    (U4, U4), (U4, U1)
    const;
    for C;
    where C: TCategoryMul<TAffine>;
    self: Transform<N, C, 3>, rhs: UnitQuaternion<N>, Output = Transform<N, C::Representative, 3>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
);

// UnitQuaternion ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (U4, U1), (U4, U4)
    const;
    for C;
    where C: TCategoryMul<TAffine>;
    self: UnitQuaternion<N>, rhs: Transform<N, C, 3>, Output = Transform<N, C::Representative, 3>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
);

//      // Transform ÷ Isometry
//      md_impl_all!(
//          Div, div where N: RealField;
//          (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
//          for Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >
//          where SB::Alloc: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1> >;
//          self: Transform<N, C, D>, rhs: Isometry<N, R, D>, Output = Transform<N, C::Representative, D>;
//          [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.inverse().to_homogeneous());
//          [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.inverse().to_homogeneous());
//          [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.inverse().to_homogeneous());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.inverse().to_homogeneous());
//      );

//      // Isometry ÷ Transform
//      md_impl_all!(
//          Div, div where N: RealField;
//          (Const<D>, U1), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
//          for Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >
//          where SA::Alloc: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1> >;
//          self: Isometry<N, R, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
//          [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//      );

//      // Transform ÷ Similarity
//      md_impl_all!(
//          Div, div where N: RealField;
//          (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
//          for Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >
//          where SB::Alloc: Allocator<N, D, D >
//          where SB::Alloc: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1> >;
//          self: Transform<N, C, D>, rhs: Similarity<N, R, D>, Output = Transform<N, C::Representative, D>;
//          [val val] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
//          [ref val] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
//          [val ref] => Self::Output::from_matrix_unchecked(self.into_inner() * rhs.to_homogeneous());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.matrix() * rhs.to_homogeneous());
//      );

//      // Similarity ÷ Transform
//      md_impl_all!(
//          Div, div where N: RealField;
//          (Const<D>, U1), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
//          for Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >
//          where SA::Alloc: Allocator<N, D, D >
//          where SA::Alloc: Allocator<N, DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1> >;
//          self: Similarity<N, R, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
//          [val val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [ref val] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.into_inner());
//          [val ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//          [ref ref] => Self::Output::from_matrix_unchecked(self.to_homogeneous() * rhs.matrix());
//      );

// Transform ÷ Translation
md_impl_all!(
    Div, div where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Transform<N, C, D>, rhs: Translation<N, D>, Output = Transform<N, C::Representative, D>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * rhs.inverse() };
);

// Translation ÷ Transform
md_impl_all!(
    Div, div where N: RealField;
    (Const<D>, U1), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategoryMul<TAffine>;
    self: Translation<N, D>, rhs: Transform<N, C, D>, Output = Transform<N, C::Representative, D>;
    [val val] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [ref val] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [val ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self.inverse() * rhs };
);

// Transform ×= Transform
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for CA, CB;
    where Const<D>: DimNameAdd<U1>, CA: TCategory, CB: SubTCategoryOf<CA>;
    self: Transform<N, CA, D>, rhs: Transform<N, CB, D>;
    [val] => *self.matrix_mut_unchecked() *= rhs.into_inner();
    [ref] => *self.matrix_mut_unchecked() *= rhs.matrix();
);

// Transform ×= Similarity
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C, R;
    where Const<D>: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
    self: Transform<N, C, D>, rhs: Similarity<N, R, D>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ×= Isometry
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C, R;
    where Const<D>: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
    self: Transform<N, C, D>, rhs: Isometry<N, R, D>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

/*
 *
 * TODO: don't explicitly build the homogeneous translation matrix.
 * Directly apply the translation, just as in `Matrix::{append,prepend}_translation`. This has not
 * been done yet because of the `DimNameDiff` requirement (which is not automatically deduced from
 * `DimNameAdd` requirement).
 *
 */
// Transform ×= Translation
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, C, D>, rhs: Translation<N, D>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ×= Rotation
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, Const<D>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, C, D>, rhs: Rotation<N, D>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ×= UnitQuaternion
md_assign_impl_all!(
    MulAssign, mul_assign where N: RealField;
    (U4, U4), (U4, U1)
    const;
    for C;
    where C: TCategory;
    self: Transform<N, C, 3>, rhs: UnitQuaternion<N>;
    [val] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
    [ref] => *self.matrix_mut_unchecked() *= rhs.to_homogeneous();
);

// Transform ÷= Transform
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>)
    const D;
    for CA, CB;
    where Const<D>: DimNameAdd<U1>, CA: SuperTCategoryOf<CB>, CB: SubTCategoryOf<TProjective>;
    self: Transform<N, CA, D>, rhs: Transform<N, CB, D>;
    [val] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.inverse() };
    [ref] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.clone().inverse() };
);

//      // Transform ÷= Similarity
//      md_assign_impl_all!(
//          DivAssign, div_assign;
//          (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
//          for Const<D>: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
//          self: Transform<N, C, D>, rhs: Similarity<N, R, D>;
//          [val] => *self *= rhs.inverse();
//          [ref] => *self *= rhs.inverse();
//      );
//
//
//      // Transform ÷= Isometry
//      md_assign_impl_all!(
//          DivAssign, div_assign;
//          (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
//          for Const<D>: DimNameAdd<U1>, C: TCategory, R: SubsetOf<MatrixN<N, DimNameSum<Const<D>, U1>> >;
//          self: Transform<N, C, D>, rhs: Isometry<N, R, D>;
//          [val] => *self *= rhs.inverse();
//          [ref] => *self *= rhs.inverse();
//      );

// Transform ÷= Translation
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, U1)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, C, D>, rhs: Translation<N, D>;
    [val] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.inverse() };
    [ref] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.inverse() };
);

// Transform ÷= Rotation
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>), (Const<D>, Const<D>)
    const D;
    for C;
    where Const<D>: DimNameAdd<U1>, C: TCategory;
    self: Transform<N, C, D>, rhs: Rotation<N, D>;
    [val] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.inverse() };
    [ref] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.inverse() };
);

// Transform ÷= UnitQuaternion
md_assign_impl_all!(
    DivAssign, div_assign where N: RealField;
    (U4, U4), (U4, U1)
    const;
    for C;
    where C: TCategory;
    self: Transform<N, C, 3>, rhs: UnitQuaternion<N>;
    [val] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.inverse() };
    [ref] => #[allow(clippy::suspicious_op_assign_impl)] { *self *= rhs.inverse() };
);
