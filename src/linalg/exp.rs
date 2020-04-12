//! This module provides the matrix exponent (exp) function to square matrices.
//!
use crate::{
    base::{
        allocator::Allocator,
        dimension::{Dim, DimMin, DimMinimum, U1},
        storage::Storage,
        DefaultAllocator,
    },
    convert, try_convert, ComplexField, MatrixN, RealField,
};

// https://github.com/scipy/scipy/blob/c1372d8aa90a73d8a52f135529293ff4edb98fc8/scipy/sparse/linalg/matfuncs.py
struct ExpmPadeHelper<N, D>
where
    N: RealField,
    D: DimMin<D>,
    DefaultAllocator: Allocator<N, D, D> + Allocator<(usize, usize), DimMinimum<D, D>>,
{
    use_exact_norm: bool,
    ident: MatrixN<N, D>,

    a: MatrixN<N, D>,
    a2: Option<MatrixN<N, D>>,
    a4: Option<MatrixN<N, D>>,
    a6: Option<MatrixN<N, D>>,
    a8: Option<MatrixN<N, D>>,
    a10: Option<MatrixN<N, D>>,

    d4_exact: Option<N>,
    d6_exact: Option<N>,
    d8_exact: Option<N>,
    d10_exact: Option<N>,

    d4_approx: Option<N>,
    d6_approx: Option<N>,
    d8_approx: Option<N>,
    d10_approx: Option<N>,
}

impl<N, D> ExpmPadeHelper<N, D>
where
    N: RealField,
    D: DimMin<D>,
    DefaultAllocator: Allocator<N, D, D> + Allocator<(usize, usize), DimMinimum<D, D>>,
{
    fn new(a: MatrixN<N, D>, use_exact_norm: bool) -> Self {
        let (nrows, ncols) = a.data.shape();
        ExpmPadeHelper {
            use_exact_norm,
            ident: MatrixN::<N, D>::identity_generic(nrows, ncols),
            a,
            a2: None,
            a4: None,
            a6: None,
            a8: None,
            a10: None,
            d4_exact: None,
            d6_exact: None,
            d8_exact: None,
            d10_exact: None,
            d4_approx: None,
            d6_approx: None,
            d8_approx: None,
            d10_approx: None,
        }
    }

    fn a2(&self) -> &MatrixN<N, D> {
        if self.a2.is_none() {
            let ap = &self.a2 as *const Option<MatrixN<N, D>> as *mut Option<MatrixN<N, D>>;
            unsafe {
                *ap = Some(&self.a * &self.a);
            }
        }
        self.a2.as_ref().unwrap()
    }

    fn a4(&self) -> &MatrixN<N, D> {
        if self.a4.is_none() {
            let ap = &self.a4 as *const Option<MatrixN<N, D>> as *mut Option<MatrixN<N, D>>;
            let a2 = self.a2();
            unsafe {
                *ap = Some(a2 * a2);
            }
        }
        self.a4.as_ref().unwrap()
    }

    fn a6(&self) -> &MatrixN<N, D> {
        if self.a6.is_none() {
            let a2 = self.a2();
            let a4 = self.a4();
            let ap = &self.a6 as *const Option<MatrixN<N, D>> as *mut Option<MatrixN<N, D>>;
            unsafe {
                *ap = Some(a4 * a2);
            }
        }
        self.a6.as_ref().unwrap()
    }

    fn a8(&self) -> &MatrixN<N, D> {
        if self.a8.is_none() {
            let a2 = self.a2();
            let a6 = self.a6();
            let ap = &self.a8 as *const Option<MatrixN<N, D>> as *mut Option<MatrixN<N, D>>;
            unsafe {
                *ap = Some(a6 * a2);
            }
        }
        self.a8.as_ref().unwrap()
    }

    fn a10(&mut self) -> &MatrixN<N, D> {
        if self.a10.is_none() {
            let a4 = self.a4();
            let a6 = self.a6();
            let ap = &self.a10 as *const Option<MatrixN<N, D>> as *mut Option<MatrixN<N, D>>;
            unsafe {
                *ap = Some(a6 * a4);
            }
        }
        self.a10.as_ref().unwrap()
    }

    fn d4_tight(&mut self) -> N {
        if self.d4_exact.is_none() {
            self.d4_exact = Some(one_norm(self.a4()).powf(convert(0.25)));
        }
        self.d4_exact.unwrap()
    }

    fn d6_tight(&mut self) -> N {
        if self.d6_exact.is_none() {
            self.d6_exact = Some(one_norm(self.a6()).powf(convert(1.0 / 6.0)));
        }
        self.d6_exact.unwrap()
    }

    fn d8_tight(&mut self) -> N {
        if self.d8_exact.is_none() {
            self.d8_exact = Some(one_norm(self.a8()).powf(convert(1.0 / 8.0)));
        }
        self.d8_exact.unwrap()
    }

    fn d10_tight(&mut self) -> N {
        if self.d10_exact.is_none() {
            self.d10_exact = Some(one_norm(self.a10()).powf(convert(1.0 / 10.0)));
        }
        self.d10_exact.unwrap()
    }

    fn d4_loose(&mut self) -> N {
        if self.use_exact_norm {
            return self.d4_tight();
        }

        if self.d4_exact.is_some() {
            return self.d4_exact.unwrap();
        }

        if self.d4_approx.is_none() {
            self.d4_approx = Some(one_norm(self.a4()).powf(convert(0.25)));
        }

        self.d4_approx.unwrap()
    }

    fn d6_loose(&mut self) -> N {
        if self.use_exact_norm {
            return self.d6_tight();
        }

        if self.d6_exact.is_some() {
            return self.d6_exact.unwrap();
        }

        if self.d6_approx.is_none() {
            self.d6_approx = Some(one_norm(self.a6()).powf(convert(1.0 / 6.0)));
        }

        self.d6_approx.unwrap()
    }

    fn d8_loose(&mut self) -> N {
        if self.use_exact_norm {
            return self.d8_tight();
        }

        if self.d8_exact.is_some() {
            return self.d8_exact.unwrap();
        }

        if self.d8_approx.is_none() {
            self.d8_approx = Some(one_norm(self.a8()).powf(convert(1.0 / 8.0)));
        }

        self.d8_approx.unwrap()
    }

    fn d10_loose(&mut self) -> N {
        if self.use_exact_norm {
            return self.d10_tight();
        }

        if self.d10_exact.is_some() {
            return self.d10_exact.unwrap();
        }

        if self.d10_approx.is_none() {
            self.d10_approx = Some(one_norm(self.a10()).powf(convert(1.0 / 10.0)));
        }

        self.d10_approx.unwrap()
    }

    fn pade3(&mut self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        let b: [N; 4] = [convert(120.0), convert(60.0), convert(12.0), convert(1.0)];
        let u = &self.a * (self.a2() * b[3] + &self.ident * b[1]);
        let v = self.a2() * b[2] + &self.ident * b[0];
        (u, v)
    }

    fn pade5(&mut self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        let b: [N; 6] = [
            convert(30240.0),
            convert(15120.0),
            convert(3360.0),
            convert(420.0),
            convert(30.0),
            convert(1.0),
        ];
        let u = &self.a * (self.a4() * b[5] + self.a2() * b[3] + &self.ident * b[1]);
        let v = self.a4() * b[4] + self.a2() * b[2] + &self.ident * b[0];
        (u, v)
    }

    fn pade7(&mut self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        let b: [N; 8] = [
            convert(17297280.0),
            convert(8648640.0),
            convert(1995840.0),
            convert(277200.0),
            convert(25200.0),
            convert(1512.0),
            convert(56.0),
            convert(1.0),
        ];
        let u =
            &self.a * (self.a6() * b[7] + self.a4() * b[5] + self.a2() * b[3] + &self.ident * b[1]);
        let v = self.a6() * b[6] + self.a4() * b[4] + self.a2() * b[2] + &self.ident * b[0];
        (u, v)
    }

    fn pade9(&mut self) -> (MatrixN<N, D>, MatrixN<N, D>) {
        let b: [N; 10] = [
            convert(17643225600.0),
            convert(8821612800.0),
            convert(2075673600.0),
            convert(302702400.0),
            convert(30270240.0),
            convert(2162160.0),
            convert(110880.0),
            convert(3960.0),
            convert(90.0),
            convert(1.0),
        ];
        let u = &self.a
            * (self.a8() * b[9]
                + self.a6() * b[7]
                + self.a4() * b[5]
                + self.a2() * b[3]
                + &self.ident * b[1]);
        let v = self.a8() * b[8]
            + self.a6() * b[6]
            + self.a4() * b[4]
            + self.a2() * b[2]
            + &self.ident * b[0];
        (u, v)
    }

    fn pade13_scaled(&mut self, s: u64) -> (MatrixN<N, D>, MatrixN<N, D>) {
        let b: [N; 14] = [
            convert(64764752532480000.0),
            convert(32382376266240000.0),
            convert(7771770303897600.0),
            convert(1187353796428800.0),
            convert(129060195264000.0),
            convert(10559470521600.0),
            convert(670442572800.0),
            convert(33522128640.0),
            convert(1323241920.0),
            convert(40840800.0),
            convert(960960.0),
            convert(16380.0),
            convert(182.0),
            convert(1.0),
        ];
        let s = s as f64;

        let mb = &self.a * convert::<f64, N>(2.0_f64.powf(-s));
        let mb2 = self.a2() * convert::<f64, N>(2.0_f64.powf(-2.0 * s));
        let mb4 = self.a4() * convert::<f64, N>(2.0.powf(-4.0 * s));
        let mb6 = self.a6() * convert::<f64, N>(2.0.powf(-6.0 * s));

        let u2 = &mb6 * (&mb6 * b[13] + &mb4 * b[11] + &mb2 * b[9]);
        let u = &mb * (&u2 + &mb6 * b[7] + &mb4 * b[5] + &mb2 * b[3] + &self.ident * b[1]);
        let v2 = &mb6 * (&mb6 * b[12] + &mb4 * b[10] + &mb2 * b[8]);
        let v = v2 + &mb6 * b[6] + &mb4 * b[4] + &mb2 * b[2] + &self.ident * b[0];
        (u, v)
    }
}

fn factorial(n: u128) -> u128 {
    if n == 1 {
        return 1;
    }
    n * factorial(n - 1)
}

/// Compute the 1-norm of a non-negative integer power of a non-negative matrix.
fn onenorm_matrix_power_nonm<N, D>(a: &MatrixN<N, D>, p: u64) -> N
where
    N: RealField,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    let nrows = a.data.shape().0;
    let mut v = crate::VectorN::<N, D>::repeat_generic(nrows, U1, convert(1.0));
    let m = a.transpose();

    for _ in 0..p {
        v = &m * v;
    }

    v.max()
}

fn ell<N, D>(a: &MatrixN<N, D>, m: u64) -> u64
where
    N: RealField,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D> + Allocator<N, D>,
{
    // 2m choose m = (2m)!/(m! * (2m-m)!)

    let a_abs_onenorm = onenorm_matrix_power_nonm(&a.abs(), 2 * m + 1);

    if a_abs_onenorm == N::zero() {
        return 0;
    }

    let choose_2m_m =
        factorial(2 * m as u128) / (factorial(m as u128) * factorial(2 * m as u128 - m as u128));
    let abs_c_recip = choose_2m_m * factorial(2 * m as u128 + 1);
    let alpha = a_abs_onenorm / one_norm(a);
    let alpha: f64 = try_convert(alpha).unwrap() / abs_c_recip as f64;

    let u = 2_f64.powf(-53.0);
    let log2_alpha_div_u = (alpha / u).log2();
    let value = (log2_alpha_div_u / (2.0 * m as f64)).ceil();
    if value > 0.0 {
        value as u64
    } else {
        0
    }
}

fn solve_p_q<N, D>(u: MatrixN<N, D>, v: MatrixN<N, D>) -> MatrixN<N, D>
where
    N: ComplexField,
    D: DimMin<D, Output = D>,
    DefaultAllocator: Allocator<N, D, D> + Allocator<(usize, usize), DimMinimum<D, D>>,
{
    let p = &u + &v;
    let q = &v - &u;

    q.lu().solve(&p).unwrap()
}

fn one_norm<N, D>(m: &MatrixN<N, D>) -> N
where
    N: RealField,
    D: Dim,
    DefaultAllocator: Allocator<N, D, D>,
{
    let mut col_sums = vec![N::zero(); m.ncols()];
    for i in 0..m.ncols() {
        let col = m.column(i);
        col.iter().for_each(|v| col_sums[i] += v.abs());
    }
    let mut max = col_sums[0];
    for i in 1..col_sums.len() {
        max = N::max(max, col_sums[i]);
    }
    max
}

impl<N: RealField, D> MatrixN<N, D>
where
    D: DimMin<D, Output = D>,
    DefaultAllocator:
        Allocator<N, D, D> + Allocator<(usize, usize), DimMinimum<D, D>> + Allocator<N, D>,
{
    /// Computes exponential of this matrix
    pub fn exp(&self) -> Self {
        // Simple case
        if self.nrows() == 1 {
            return self.clone().map(|v| v.exp());
        }

        let mut h = ExpmPadeHelper::new(self.clone(), true);

        let eta_1 = N::max(h.d4_loose(), h.d6_loose());
        if eta_1 < convert(1.495585217958292e-002) && ell(&h.a, 3) == 0 {
            let (u, v) = h.pade3();
            return solve_p_q(u, v);
        }

        let eta_2 = N::max(h.d4_tight(), h.d6_loose());
        if eta_2 < convert(2.539398330063230e-001) && ell(&h.a, 5) == 0 {
            let (u, v) = h.pade5();
            return solve_p_q(u, v);
        }

        let eta_3 = N::max(h.d6_tight(), h.d8_loose());
        if eta_3 < convert(9.504178996162932e-001) && ell(&h.a, 7) == 0 {
            let (u, v) = h.pade7();
            return solve_p_q(u, v);
        }
        if eta_3 < convert(2.097847961257068e+000) && ell(&h.a, 9) == 0 {
            let (u, v) = h.pade9();
            return solve_p_q(u, v);
        }

        let eta_4 = N::max(h.d8_loose(), h.d10_loose());
        let eta_5 = N::min(eta_3, eta_4);
        let theta_13 = convert(4.25);

        let mut s = if eta_5 == N::zero() {
            0
        } else {
            let l2 = try_convert((eta_5 / theta_13).log2().ceil()).unwrap();

            if l2 < 0.0 {
                0
            } else {
                l2 as u64
            }
        };

        s += ell(&(&h.a * convert::<f64, N>(2.0_f64.powf(-(s as f64)))), 13);

        let (u, v) = h.pade13_scaled(s);
        let mut x = solve_p_q(u, v);

        for _ in 0..s {
            x = &x * &x;
        }
        x
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn one_norm() {
        use crate::Matrix3;
        let m = Matrix3::new(-3.0, 5.0, 7.0, 2.0, 6.0, 4.0, 0.0, 2.0, 8.0);

        assert_eq!(super::one_norm(&m), 19.0);
    }
}
