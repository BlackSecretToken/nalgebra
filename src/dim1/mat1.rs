use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::division_ring::DivisionRing;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::transpose::Transpose;
use traits::transformation::Transform; // FIXME: implement Transformable, Transformation
use traits::rlmul::{RMul, LMul};
use vec::Vec1;

#[deriving(Eq, ToStr)]
pub struct Mat1<N>
{ m11: N }

impl<N> Mat1<N>
{
  #[inline]
  pub fn new(m11: N) -> Mat1<N>
  {
    Mat1
    { m11: m11 }
  }
}

impl<N> Dim for Mat1<N>
{
  #[inline]
  fn dim() -> uint
  { 1 }
}

impl<N: One> One for Mat1<N>
{
  #[inline]
  fn one() -> Mat1<N>
  { return Mat1::new(One::one()) }
}

impl<N: Zero> Zero for Mat1<N>
{
  #[inline]
  fn zero() -> Mat1<N>
  { Mat1::new(Zero::zero()) }

  #[inline]
  fn is_zero(&self) -> bool
  { self.m11.is_zero() }
}

impl<N: Mul<N, N> + Add<N, N>> Mul<Mat1<N>, Mat1<N>> for Mat1<N>
{
  #[inline]
  fn mul(&self, other: &Mat1<N>) -> Mat1<N>
  { Mat1::new(self.m11 * other.m11) }
}

impl<N: Copy + DivisionRing>
Transform<Vec1<N>> for Mat1<N>
{
  #[inline]
  fn transform_vec(&self, v: &Vec1<N>) -> Vec1<N>
  { self.rmul(v) }

  #[inline]
  fn inv_transform(&self, v: &Vec1<N>) -> Vec1<N>
  { self.inverse().transform_vec(v) }
}

impl<N: Add<N, N> + Mul<N, N>> RMul<Vec1<N>> for Mat1<N>
{
  #[inline]
  fn rmul(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new([self.m11 * other.at[0]]) }
}

impl<N: Add<N, N> + Mul<N, N>> LMul<Vec1<N>> for Mat1<N>
{
  #[inline]
  fn lmul(&self, other: &Vec1<N>) -> Vec1<N>
  { Vec1::new([self.m11 * other.at[0]]) }
}

impl<N: Copy + DivisionRing>
Inv for Mat1<N>
{
  #[inline]
  fn inverse(&self) -> Mat1<N>
  {
    let mut res : Mat1<N> = copy *self;

    res.invert();

    res
  }

  #[inline]
  fn invert(&mut self)
  {
    assert!(!self.m11.is_zero());

    self.m11 = One::one::<N>() / self.m11
  }
}

impl<N: Copy> Transpose for Mat1<N>
{
  #[inline]
  fn transposed(&self) -> Mat1<N>
  { copy *self }

  #[inline]
  fn transpose(&mut self)
  { }
}

impl<N: ApproxEq<N>> ApproxEq<N> for Mat1<N>
{
  #[inline]
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  #[inline]
  fn approx_eq(&self, other: &Mat1<N>) -> bool
  { self.m11.approx_eq(&other.m11) }

  #[inline]
  fn approx_eq_eps(&self, other: &Mat1<N>, epsilon: &N) -> bool
  { self.m11.approx_eq_eps(&other.m11, epsilon) }
}

impl<N: Rand> Rand for Mat1<N>
{
  #[inline]
  fn rand<R: Rng>(rng: &mut R) -> Mat1<N>
  { Mat1::new(rng.gen()) }
}
