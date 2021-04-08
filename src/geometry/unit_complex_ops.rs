use std::ops::{Div, DivAssign, Mul, MulAssign};

use crate::base::storage::Storage;
use crate::base::{Const, Unit, Vector, Vector2};
use crate::geometry::{Isometry, Point2, Rotation, Similarity, Translation, UnitComplex};
use simba::simd::SimdRealField;

/*
 * This file provides:
 * ===================
 *
 * UnitComplex  × UnitComplex
 * UnitComplex  × Rotation -> UnitComplex
 * Rotation × UnitComplex  -> UnitComplex
 *
 * UnitComplex  ÷ UnitComplex
 * UnitComplex  ÷ Rotation -> UnitComplex
 * Rotation ÷ UnitComplex  -> UnitComplex
 *
 *
 * UnitComplex × Point
 * UnitComplex × Vector
 * UnitComplex × Unit<T>
 *
 * UnitComplex × Isometry<UnitComplex>
 * UnitComplex × Similarity<UnitComplex>
 * UnitComplex × Translation -> Isometry<UnitComplex>
 *
 * (Assignment Operators)
 *
 * UnitComplex  ×= UnitComplex
 * UnitComplex  ×= Rotation
 *
 * UnitComplex  ÷= UnitComplex
 * UnitComplex  ÷= Rotation
 *
 * Rotation ×= UnitComplex
 * Rotation ÷= UnitComplex
 *
 */

// UnitComplex × UnitComplex
impl<N: SimdRealField> Mul<Self> for UnitComplex<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Unit::new_unchecked(self.into_inner() * rhs.into_inner())
    }
}

impl<'a, N: SimdRealField> Mul<UnitComplex<N>> for &'a UnitComplex<N>
where
    N::Element: SimdRealField,
{
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.complex() * rhs.into_inner())
    }
}

impl<'b, N: SimdRealField> Mul<&'b UnitComplex<N>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.into_inner() * rhs.as_ref())
    }
}

impl<'a, 'b, N: SimdRealField> Mul<&'b UnitComplex<N>> for &'a UnitComplex<N>
where
    N::Element: SimdRealField,
{
    type Output = UnitComplex<N>;

    #[inline]
    fn mul(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        Unit::new_unchecked(self.complex() * rhs.as_ref())
    }
}

// UnitComplex ÷ UnitComplex
impl<N: SimdRealField> Div<Self> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        Unit::new_unchecked(self.into_inner() * rhs.conjugate().into_inner())
    }
}

impl<'a, N: SimdRealField> Div<UnitComplex<N>> for &'a UnitComplex<N>
where
    N::Element: SimdRealField,
{
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: UnitComplex<N>) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        Unit::new_unchecked(self.complex() * rhs.conjugate().into_inner())
    }
}

impl<'b, N: SimdRealField> Div<&'b UnitComplex<N>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        Unit::new_unchecked(self.into_inner() * rhs.conjugate().into_inner())
    }
}

impl<'a, 'b, N: SimdRealField> Div<&'b UnitComplex<N>> for &'a UnitComplex<N>
where
    N::Element: SimdRealField,
{
    type Output = UnitComplex<N>;

    #[inline]
    fn div(self, rhs: &'b UnitComplex<N>) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        Unit::new_unchecked(self.complex() * rhs.conjugate().into_inner())
    }
}

macro_rules! complex_op_impl(
    ($Op: ident, $op: ident;
     $($Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     $action: expr; $($lives: tt),*) => {
        impl<$($lives ,)* N: SimdRealField $(, $Storage: $StoragesBound $(<$($BoundParam),*>)*)*> $Op<$Rhs> for $Lhs
            where N::Element: SimdRealField {
            type Output = $Result;

            #[inline]
            fn $op($lhs, $rhs: $Rhs) -> Self::Output {
                $action
            }
        }
    }
);

macro_rules! complex_op_impl_all(
    ($Op: ident, $op: ident;
     $($Storage: ident: $StoragesBound: ident $(<$($BoundParam: ty),*>)*),*;
     $lhs: ident: $Lhs: ty, $rhs: ident: $Rhs: ty, Output = $Result: ty;
     [val val] => $action_val_val: expr;
     [ref val] => $action_ref_val: expr;
     [val ref] => $action_val_ref: expr;
     [ref ref] => $action_ref_ref: expr;) => {

    complex_op_impl!($Op, $op;
                     $($Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_val_val; );

    complex_op_impl!($Op, $op;
                     $($Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: &'a $Lhs, $rhs: $Rhs, Output = $Result;
                     $action_ref_val; 'a);

    complex_op_impl!($Op, $op;
                     $($Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_val_ref; 'b);

    complex_op_impl!($Op, $op;
                     $($Storage: $StoragesBound $(<$($BoundParam),*>)*),*;
                     $lhs: &'a $Lhs, $rhs: &'b $Rhs, Output = $Result;
                     $action_ref_ref; 'a, 'b);


    }
);

// UnitComplex × Rotation
complex_op_impl_all!(
    Mul, mul;
    ;
    self: UnitComplex<N>, rhs: Rotation<N, 2>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] =>  self * UnitComplex::from_rotation_matrix(rhs);
);

// UnitComplex ÷ Rotation
complex_op_impl_all!(
    Div, div;
    ;
    self: UnitComplex<N>, rhs: Rotation<N, 2>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { self * UnitComplex::from_rotation_matrix(rhs).inverse() };
);

// Rotation × UnitComplex
complex_op_impl_all!(
    Mul, mul;
    ;
    self: Rotation<N, 2>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => UnitComplex::from_rotation_matrix(self) * rhs;
);

// Rotation ÷ UnitComplex
complex_op_impl_all!(
    Div, div;
    ;
    self: Rotation<N, 2>, rhs: UnitComplex<N>, Output = UnitComplex<N>;
    [val val] => &self / &rhs;
    [ref val] =>  self / &rhs;
    [val ref] => &self /  rhs;
    [ref ref] => #[allow(clippy::suspicious_arithmetic_impl)] { UnitComplex::from_rotation_matrix(self) * rhs.inverse() };
);

// UnitComplex × Point
complex_op_impl_all!(
    Mul, mul;
    ;
    self: UnitComplex<N>, rhs: Point2<N>, Output = Point2<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Point2::from(self * &rhs.coords);
);

// UnitComplex × Vector
complex_op_impl_all!(
    Mul, mul;
    S: Storage<N, Const<2>>;
    self: UnitComplex<N>, rhs: Vector<N, Const<2>, S>, Output = Vector2<N>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let i = self.as_ref().im;
        let r = self.as_ref().re;
        Vector2::new(r * rhs[0] - i * rhs[1], i * rhs[0] + r * rhs[1])
    };
);

// UnitComplex × Unit<Vector>
complex_op_impl_all!(
    Mul, mul;
    S: Storage<N, Const<2>>;
    self: UnitComplex<N>, rhs: Unit<Vector<N, Const<2>, S>>, Output = Unit<Vector2<N>>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Unit::new_unchecked(self * rhs.as_ref());
);

// UnitComplex × Isometry<UnitComplex>
complex_op_impl_all!(
    Mul, mul;
    ;
    self: UnitComplex<N>, rhs: Isometry<N, UnitComplex<N>, 2>,
    Output = Isometry<N, UnitComplex<N>, 2>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => {
        let shift = self * &rhs.translation.vector;
        Isometry::from_parts(Translation::from(shift), self * &rhs.rotation)
    };
);

// UnitComplex × Similarity<UnitComplex>
complex_op_impl_all!(
    Mul, mul;
    ;
    self: UnitComplex<N>, rhs: Similarity<N, UnitComplex<N>, 2>,
    Output = Similarity<N, UnitComplex<N>, 2>;
    [val val] => &self * &rhs;
    [ref val] =>  self * &rhs;
    [val ref] => &self *  rhs;
    [ref ref] => Similarity::from_isometry(self * &rhs.isometry, rhs.scaling());
);

// UnitComplex × Translation
complex_op_impl_all!(
    Mul, mul;
    ;
    self: UnitComplex<N>, rhs: Translation<N, 2>,
    Output = Isometry<N, UnitComplex<N>, 2>;
    [val val] => Isometry::from_parts(Translation::from(&self *  rhs.vector), self);
    [ref val] => Isometry::from_parts(Translation::from( self *  rhs.vector), *self);
    [val ref] => Isometry::from_parts(Translation::from(&self * &rhs.vector), self);
    [ref ref] => Isometry::from_parts(Translation::from( self * &rhs.vector), *self);
);

// Translation × UnitComplex
complex_op_impl_all!(
    Mul, mul;
    ;
    self: Translation<N, 2>, right: UnitComplex<N>,
    Output = Isometry<N, UnitComplex<N>, 2>;
    [val val] => Isometry::from_parts(self, right);
    [ref val] => Isometry::from_parts(self.clone(), right);
    [val ref] => Isometry::from_parts(self, *right);
    [ref ref] => Isometry::from_parts(self.clone(), *right);
);

// UnitComplex ×= UnitComplex
impl<N: SimdRealField> MulAssign<UnitComplex<N>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn mul_assign(&mut self, rhs: UnitComplex<N>) {
        *self = &*self * rhs
    }
}

impl<'b, N: SimdRealField> MulAssign<&'b UnitComplex<N>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b UnitComplex<N>) {
        *self = &*self * rhs
    }
}

// UnitComplex /= UnitComplex
impl<N: SimdRealField> DivAssign<UnitComplex<N>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn div_assign(&mut self, rhs: UnitComplex<N>) {
        *self = &*self / rhs
    }
}

impl<'b, N: SimdRealField> DivAssign<&'b UnitComplex<N>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn div_assign(&mut self, rhs: &'b UnitComplex<N>) {
        *self = &*self / rhs
    }
}

// UnitComplex ×= Rotation
impl<N: SimdRealField> MulAssign<Rotation<N, 2>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Rotation<N, 2>) {
        *self = &*self * rhs
    }
}

impl<'b, N: SimdRealField> MulAssign<&'b Rotation<N, 2>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b Rotation<N, 2>) {
        *self = &*self * rhs
    }
}

// UnitComplex ÷= Rotation
impl<N: SimdRealField> DivAssign<Rotation<N, 2>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn div_assign(&mut self, rhs: Rotation<N, 2>) {
        *self = &*self / rhs
    }
}

impl<'b, N: SimdRealField> DivAssign<&'b Rotation<N, 2>> for UnitComplex<N>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn div_assign(&mut self, rhs: &'b Rotation<N, 2>) {
        *self = &*self / rhs
    }
}

// Rotation ×= UnitComplex
impl<N: SimdRealField> MulAssign<UnitComplex<N>> for Rotation<N, 2>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn mul_assign(&mut self, rhs: UnitComplex<N>) {
        self.mul_assign(rhs.to_rotation_matrix())
    }
}

impl<'b, N: SimdRealField> MulAssign<&'b UnitComplex<N>> for Rotation<N, 2>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'b UnitComplex<N>) {
        self.mul_assign(rhs.to_rotation_matrix())
    }
}

// Rotation ÷= UnitComplex
impl<N: SimdRealField> DivAssign<UnitComplex<N>> for Rotation<N, 2>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn div_assign(&mut self, rhs: UnitComplex<N>) {
        self.div_assign(rhs.to_rotation_matrix())
    }
}

impl<'b, N: SimdRealField> DivAssign<&'b UnitComplex<N>> for Rotation<N, 2>
where
    N::Element: SimdRealField,
{
    #[inline]
    fn div_assign(&mut self, rhs: &'b UnitComplex<N>) {
        self.div_assign(rhs.to_rotation_matrix())
    }
}
