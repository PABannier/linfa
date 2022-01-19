// Have a vanilla version of LARS work (make it work)
// Bring speed optimization (pre-compute Gram matrix and Xy)

// input: X, y, max_iter, alpha_min, verbose, eps
// output: active_set (indices of active features at the end of the path)
// output: signs (coefficients of the active features along the path, 0 if not active)
// output: alphas: maximum of variances along the path

extern crate ndarray;

use ndarray::linalg::general_mat_mul;
use ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Dimension, OwnedRepr, RemoveAxis,
};
use ndarray_linalg::{
    cholesky::SolveC,
    solveh::UPLO,
    triangular::{Diag, SolveTriangular},
    Lapack,
};
use num_traits::Float;
use std::ops::Mul;

fn lars<'a, F: 'static + Float + Lapack + Ord>(
    x: ArrayView2<'a, F>,
    y: ArrayView1<'a, F>,
    max_iterations: usize,
    alpha_min: F,
    eps: F,
    verbose: bool,
) -> (Array1<F>, Vec<usize>, Array2<F>) {
    // Define input
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    let mut cov = x.t().dot(&y);
    let mut gram = Array2::<F>::zeros((n_features, n_features));
    general_mat_mul(F::one(), &x.t(), &x, F::one(), &mut gram);

    // Future optimization: for faster memory access, make X in fortran order

    let max_features = usize::min(max_iterations, n_features);

    let mut coefficients = Array2::<F>::zeros((max_features + 1, n_features));
    let alphas = Array1::<F>::zeros(max_features + 1);

    let mut n_iter = 0;
    let mut n_active = 0;

    let mut active = Vec::new();
    // holds the active set
    let mut indices = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();
    // holds the sign of covariance
    let mut sign_active = Array1::<F>::zeros(max_features);
    let mut drop = false;
    // cholesky factorization of the Gram matrix (lower part)
    let mut cholesky_lower = Array2::<F>::zeros((max_features, max_features));

    loop {
        // takes argmax of covariance
        let (c_idx, &largest_abs_cov) = cov
            .map(|&x| <F as Float>::abs(x))
            .as_slice()
            .unwrap()
            .iter()
            .enumerate()
            .max_by_key(|(_, &value)| value)
            .unwrap();

        let largest_cov = cov[c_idx];

        let mut alpha = alphas[n_iter];
        let prev_alpha = alphas[n_iter - 1];

        let mut coef = coefficients.slice(s![n_iter, ..]).to_owned();
        let prev_coef = coefficients.slice(s![n_iter - 1, ..]);

        alpha = largest_abs_cov / F::from(n_samples).unwrap();
        // early stopping
        if alpha <= alpha_min + F::epsilon() {
            if <F as Float>::abs(alpha - alpha_min) > F::epsilon() {
                // interpolation factor 0 <= ss < 1
                if n_iter > 0 {
                    // in the first iteration, all alphas are zero, the formula below
                    // would make ss a NaN
                    // ss is the scaling factor to make equivariant
                    let ss = (prev_alpha - alpha_min) / (prev_alpha - alpha);
                    // one step in the direction of the x_j feature
                    for j in 0..n_features {
                        coef[j] = prev_coef[j] + ss * (coef[j] - prev_coef[j]);
                    }
                }
                alpha = alpha_min;
                coefficients.slice_mut(s![n_iter, ..]).assign(&coef);
                break;
            }
        }
        if n_iter >= max_iterations || n_active >= n_features {
            break;
        }
        if !drop {
            sign_active[n_active] = F::signum(largest_cov);
            let m = n_active;
            let n = c_idx + n_active;

            swap_elements(&mut cov, (0, c_idx), 0);
            swap_elements(&mut indices, (m, n), 0);
            let mut cov = cov.slice(s![1..]).to_owned(); // remove cov[0]

            swap_elements(&mut gram, (m, n), 0);
            swap_elements(&mut gram, (m, n), 1);
            let c = gram[[n_active, n_active]];
            // looping through elements seems faster than slice and assign
            for j in 0..n_active {
                cholesky_lower[[n_active, j]] = gram[[n_active, j]];
            }

            // update the cholesky decomposition for the Gram matrix
            if n_active > 0 {
                // this copy might be expensive...
                let b = cholesky_lower.slice(s![n_active, ..n_active]).to_owned();
                let _res = cholesky_lower
                    .slice(s![..n_active, ..n_active])
                    .solve_triangular(UPLO::Lower, Diag::NonUnit, &b)
                    .expect("Cholesky matrix is non-singular");
                cholesky_lower
                    .slice_mut(s![n_active, ..n_active])
                    .assign(&_res);
            }

            let v = cholesky_lower
                .slice(s![n_active, ..n_active])
                .dot(&cholesky_lower.slice(s![n_active, ..n_active]));
            let diag = <F as Float>::max(<F as Float>::sqrt(<F as Float>::abs(c - v)), eps);
            cholesky_lower[[n_active, n_active]] = diag;

            active.push(indices[n_active]);
            n_active += 1;

            if verbose {
                println!(
                    "n_iter: {} :: active_idx: {} :: n_active: {} :: cov: {}",
                    n_iter,
                    active[active.len() - 1],
                    n_active,
                    largest_abs_cov
                );
            }

            // least square solution
            let sign_active_restricted = sign_active.slice(s![..n_active]);
            let mut least_squares = cholesky_lower
                .slice(s![..n_active, ..n_active])
                .solvec(&sign_active_restricted)
                .expect("Cholesky matrix is not singular");

            let aa;
            if least_squares.len() == 1 && least_squares[0] == F::zero() {
                // this happens because sign_active[:n_active] = 0
                least_squares[0] = F::one();
                aa = F::one();
            } else {
                aa = F::one()
                    / <F as Float>::sqrt(
                        least_squares.mul(sign_active.slice(s![..n_active])).sum(),
                    );
                // if !F::is_finite(aa) {
                //     // L is too ill-conditioned
                //     let i = 0;
                //     let cholesky_lower_ = cholesky_lower.slice(s![..n_active, ..n_active]).to_owned();
                //     while !F::is_finite(aa) {
                //         // TODO: L_.flat[:: n_active + 1] += (2 ** i) *
                //         let sign_active_restricted = sign_active.slice(s![..n_active]);
                //         let least_squares = L_.solvec(&sign_active_restricted)
                //                               .expect("Cholesky matrix is not singular");
                //         let tmp = <F as Float>::max(
                //             F::from(least_squares.mul(sign_active.slice(s![..n_active])).sum()).unwrap(),
                //             F::epsilon()
                //         );
                //         aa = F::one() / <F as Float>::sqrt(tmp);
                //         i += 1;
                //     }
                // }
                for j in 0..n_active {
                    least_squares[j] *= aa;
                }
            }

            // very time-consuming; could be enhanced by taking the QR decomposition of x
            let corr_eq_dir = gram
                .slice(s![..n_active, n_active..])
                .t()
                .dot(&least_squares); // shape: (n_features - n_active,)

            // avoid unstable results because of rounding errors
            // let corr_eq_dir =
            //     corr_eq_dir.map(|&x| F::trunc(x * F::from(1e8).unwrap()) / F::from(1e8).unwrap());

            // min-pos
            let mut g1 = F::infinity();
            for j in 0..n_features {
                let tmp = (largest_cov - cov[j]) / (aa - corr_eq_dir[j] + F::epsilon()); // TODO: check corr_eq_dir
                if g1 > tmp && tmp >= F::zero() {
                    g1 = tmp;
                }
            }

            // min-pos
            let mut g2 = F::infinity();
            for j in 0..n_features {
                let tmp = (largest_cov + cov[j]) / (aa + corr_eq_dir[j] + F::epsilon()); // TODO: check corr_eq_dir
                if g2 > tmp && tmp >= F::zero() {
                    g2 = tmp;
                }
            }
            let gamma_ = <F as Float>::min(g1, <F as Float>::min(g2, largest_cov / aa));

            drop = false;
            let mut z = Array1::<F>::zeros(n_features);
            let mut z_pos = F::infinity();
            for (idx, &j) in active.iter().enumerate() {
                z[j] = -coef[j] / (least_squares[idx] + F::epsilon());
                if z[j] >= F::zero() && z[j] < z_pos {
                    z_pos = z[j];
                }
            }
            if z_pos < gamma_ {
                // some coefficients have changed sign
                let mut indices = Vec::new();
                for (idx, &j) in z.iter().enumerate() {
                    if j == z_pos {
                        indices.push(idx);
                    }
                }

                // update the sign
                for &j in indices.iter() {
                    sign_active[j] = -sign_active[j];
                }
                drop = true;
            }

            n_iter += 1;

            if n_iter >= coefficients.shape()[0] {
                std::mem::drop(coef);
                std::mem::drop(alpha);
                std::mem::drop(prev_alpha);
                std::mem::drop(prev_coef);
                // resize the coefficients and alphas arrays
                let add_features = 2 * usize::max(1, max_features - n_active);
                let mut coefficients = coefficients
                    .into_shape((n_iter + add_features, n_features))
                    .expect("coefficients have the correct shape");
                for i in add_features..n_iter + add_features {
                    for j in 0..n_features {
                        coefficients[[i, j]] = F::zero();
                    }
                }
                let mut alphas = alphas
                    .into_shape(n_iter + add_features)
                    .expect("alphas have the correct shape");
                for i in add_features..n_iter + add_features {
                    alphas[i] = F::zero();
                }
            }
            let mut coef = coefficients.slice(s![n_iter, ..]).to_owned();
            let prev_coef = coefficients.slice(s![n_iter - 1, ..]);

            // Making a forward step
            for (idx, &j) in active.iter().enumerate() {
                coef[j] = prev_coef[j] + gamma_ * least_squares[idx];
            }

            // Update correlations
            for j in 0..n_features {
                cov[j] -= gamma_ * corr_eq_dir[j]; // TODO: check dimension of corr_eq_dir
            }
        }
    }
    // Resize coefficients in case of early stopping
    let out_alphas = alphas.slice(s![..n_iter + 1]).to_owned();
    let out_coefficients = coefficients.slice(s![..n_iter + 1, ..]).t().to_owned();

    (out_alphas, active, out_coefficients)
}

fn swap_elements<F, I>(x: &mut ArrayBase<OwnedRepr<F>, I>, perm: (usize, usize), axis: usize)
where
    I: Dimension + RemoveAxis,
{
    let mut it = x.axis_iter_mut(Axis(axis));

    it.nth(perm.0)
        .unwrap()
        .into_iter()
        .zip(it.nth(perm.1 - (perm.0 + 1)).unwrap().into_iter())
        .map(|(mut x, mut y)| std::mem::swap(&mut x, &mut y));
}
