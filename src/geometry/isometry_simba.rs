use simba::simd::SimdValue;

use crate::SimdRealField;

use crate::geometry::{AbstractRotation, Isometry, Translation};

impl<N: SimdRealField, R, const D: usize> SimdValue for Isometry<N, R, D>
where
    N::Element: SimdRealField,
    R: SimdValue<SimdBool = N::SimdBool> + AbstractRotation<N, D>,
    R::Element: AbstractRotation<N::Element, D>,
{
    type Element = Isometry<N::Element, R::Element, D>;
    type SimdBool = N::SimdBool;

    #[inline]
    fn lanes() -> usize {
        N::lanes()
    }

    #[inline]
    fn splat(val: Self::Element) -> Self {
        Isometry::from_parts(Translation::splat(val.translation), R::splat(val.rotation))
    }

    #[inline]
    fn extract(&self, i: usize) -> Self::Element {
        Isometry::from_parts(self.translation.extract(i), self.rotation.extract(i))
    }

    #[inline]
    unsafe fn extract_unchecked(&self, i: usize) -> Self::Element {
        Isometry::from_parts(
            self.translation.extract_unchecked(i),
            self.rotation.extract_unchecked(i),
        )
    }

    #[inline]
    fn replace(&mut self, i: usize, val: Self::Element) {
        self.translation.replace(i, val.translation);
        self.rotation.replace(i, val.rotation);
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, i: usize, val: Self::Element) {
        self.translation.replace_unchecked(i, val.translation);
        self.rotation.replace_unchecked(i, val.rotation);
    }

    #[inline]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        Isometry::from_parts(
            self.translation.select(cond, other.translation),
            self.rotation.select(cond, other.rotation),
        )
    }
}
