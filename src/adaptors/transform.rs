use std::num::{One, Zero};
use std::rand::{Rand, Rng, RngUtil};
use std::cmp::ApproxEq;
use traits::dim::Dim;
use traits::inv::Inv;
use traits::rotation::Rotation;
use traits::translation::Translation;
use traits::transpose::Transpose;
use traits::delta_transform::{DeltaTransform, DeltaTransformVector};
use traits::workarounds::rlmul::{RMul, LMul};

#[deriving(Eq, ToStr)]
pub struct Transform<M, V>
{
  priv submat   : M,
  priv subtrans : V
}

pub fn transform<M: Copy, V: Copy>(mat: &M, trans: &V)
-> Transform<M, V>
{ Transform { submat: *mat, subtrans: *trans } }

impl<M:Dim, V> Dim for Transform<M, V>
{
  fn dim() -> uint
  { Dim::dim::<M>() }
}

impl<M:Copy + One, V:Copy + Zero> One for Transform<M, V>
{
  fn one() -> Transform<M, V>
  { Transform { submat: One::one(), subtrans: Zero::zero() } }
}

impl<M:Copy + Zero, V:Copy + Zero> Zero for Transform<M, V>
{
  fn zero() -> Transform<M, V>
  { Transform { submat: Zero::zero(), subtrans: Zero::zero() } }

  fn is_zero(&self) -> bool
  { self.submat.is_zero() && self.subtrans.is_zero() }
}

impl<M:Copy + RMul<V> + Mul<M, M>, V:Copy + Add<V, V>>
Mul<Transform<M, V>, Transform<M, V>> for Transform<M, V>
{
  fn mul(&self, other: &Transform<M, V>) -> Transform<M, V>
  {
    Transform { submat: self.submat * other.submat,
                subtrans: self.subtrans + self.submat.rmul(&other.subtrans) }
  }
}

impl<M: RMul<V>, V: Add<V, V>> RMul<V> for Transform<M, V>
{
  fn rmul(&self, other: &V) -> V
  { self.submat.rmul(other) + self.subtrans }
}

impl<M: LMul<V>, V: Add<V, V>> LMul<V> for Transform<M, V>
{
  fn lmul(&self, other: &V) -> V
  { self.submat.lmul(other) + self.subtrans }
}

impl<M: Copy, V: Copy + Translation<V>> Translation<V> for Transform<M, V>
{
  fn translation(&self) -> V
  { self.subtrans.translation() }

  fn translated(&self, t: &V) -> Transform<M, V>
  { transform(&self.submat, &self.subtrans.translated(t)) }

  fn translate(&mut self, t: &V)
  { self.subtrans.translate(t) }
}

impl<M: Rotation<AV> + Copy + RMul<V> + One, V: Copy, AV>
Rotation<AV> for Transform<M, V>
{
  fn rotation(&self) -> AV
  { self.submat.rotation() }

  fn rotated(&self, rot: &AV) -> Transform<M, V>
  {
    // FIXME: this does not seem opitmal
    let delta = One::one::<M>().rotated(rot);

    transform(&self.submat.rotated(rot), &delta.rmul(&self.subtrans))
  }

  fn rotate(&mut self, rot: &AV)
  {
    // FIXME: this does not seem opitmal
    let delta = One::one::<M>().rotated(rot);
    self.submat.rotate(rot);
    self.subtrans = delta.rmul(&self.subtrans);
  }
}

impl<M: Copy, V> DeltaTransform<M> for Transform<M, V>
{
  fn delta_transform(&self) -> M
  { self.submat }
}

impl<M: RMul<V> + Copy, V> DeltaTransformVector<V> for Transform<M, V>
{
  fn delta_transform_vector(&self, v: &V) -> V
  { self.submat.rmul(v) }
}

impl<M:Copy + Transpose + Inv + RMul<V>, V:Copy + Neg<V>>
Inv for Transform<M, V>
{
  fn invert(&mut self)
  {
    self.submat.invert();
    self.subtrans = self.submat.rmul(&-self.subtrans);
  }

  fn inverse(&self) -> Transform<M, V>
  {
    let mut res = *self;

    res.invert();

    res
  }
}

impl<N: ApproxEq<N>, M:ApproxEq<N>, V:ApproxEq<N>>
ApproxEq<N> for Transform<M, V>
{
  fn approx_epsilon() -> N
  { ApproxEq::approx_epsilon::<N, N>() }

  fn approx_eq(&self, other: &Transform<M, V>) -> bool
  {
    self.submat.approx_eq(&other.submat) &&
    self.subtrans.approx_eq(&other.subtrans)
  }

  fn approx_eq_eps(&self, other: &Transform<M, V>, epsilon: &N) -> bool
  {
    self.submat.approx_eq_eps(&other.submat, epsilon) &&
    self.subtrans.approx_eq_eps(&other.subtrans, epsilon)
  }
}

impl<M: Rand + Copy, V: Rand + Copy> Rand for Transform<M, V>
{
  fn rand<R: Rng>(rng: &mut R) -> Transform<M, V>
  { transform(&rng.gen(), &rng.gen()) }
}
