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

fn lars<F: 'static + Float + Lapack + Ord>(
    x: ArrayView2<F>,
    y: ArrayView1<F>,
    max_iterations: usize,
    alpha_min: F,
    eps: F,
    verbose: bool,
) {
    // Define input
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    let mut cov = x.t().dot(&y);
    let mut gram = Array2::<F>::zeros((n_features, n_features));
    general_mat_mul(F::one(), &x.t(), &x, F::one(), &mut gram);

    // Future optimization: for faster memory access, make X in fortran order

    let max_features = usize::min(max_iterations, n_features);

    let mut coefficients = Array2::<F>::zeros((max_features + 1, n_features));
    let mut alphas = Array1::<F>::zeros(max_features + 1);

    let mut n_iter = 0;
    let mut n_active = 0;

    let mut active = Vec::new();
    // holds the active set
    let mut indices = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();
    // holds the sign of covariance
    let mut sign_active = Array1::<F>::zeros(max_features);
    let drop = false;
    // cholesky factorization of the Gram matrix (lower part)
    let mut cholesky_lower = Array2::<F>::zeros((max_features, max_features));

    // make copy for further computations
    let gram_copy = gram.to_owned();
    let cov_copy = cov.to_owned();

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

        let mut alpha = Array1::from_shape_vec(1, vec![alphas[n_iter]]).unwrap();
        let mut coef = coefficients.slice(s![n_iter, ..]).to_owned();
        let prev_alpha = Array1::from_shape_vec(1, vec![alphas[n_iter - 1]]).unwrap();
        let prev_coef = coefficients.slice(s![n_iter - 1, ..]);

        alpha[0] = largest_abs_cov / F::from(n_samples).unwrap();
        // early stopping
        if alpha[0] <= alpha_min + F::epsilon() {
            if <F as Float>::abs(alpha[0] - alpha_min) > F::epsilon() {
                // interpolation factor 0 <= ss < 1
                if n_iter > 0 {
                    // in the first iteration, all alphas are zero, the formula below
                    // would make ss a NaN
                    // ss is the scaling factor to make equivariant
                    let ss = (prev_alpha[0] - alpha_min) / (prev_alpha[0] - alpha[0]);
                    // one step in the direction of the x_j feature
                    for j in 0..n_features {
                        coef[j] = prev_coef[j] + ss * (coef[j] - prev_coef[j]);
                    }
                }
                alpha[0] = alpha_min;
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
            let cov_not_shortened = cov.to_owned();
            let cov = cov.slice(s![1..]); // remove cov[0]

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
            let least_squares = cholesky_lower
                .slice(s![..n_active, ..n_active])
                .solvec(&sign_active_restricted)
                .expect("Cholesky matrix is not singular");

            let mut AA;
            if least_squares.len() == 1 && least_squares[0] == F::zero() {
                // this happens because sign_active[:n_active] = 0
                least_squares[0] = 1;
                AA = F::one();
            } else {
                AA = F::one() / F::sqrt(F::sum(????)));

                if !F::is_finite(AA) {
                    // L is too ill-conditioned
                    let i = 0;
                    let cholesky_lower_ = cholesky_lower.slice(s![..n_active, ..n_active]).to_owned();
                    while F::is_infinite() {
                        ??????
                    }
                }
                least_squares *= AA;
            }

            // very time-consuming; could be enhanced by taking the QR decomposition of x
            let corr_eq_dir = gram.slice(s![..n_active, n_active..]).t().dot(&least_squares);

            // avoid unstable results because of rounding errors
            let corr_eq_dir = F::trunc(corr_eq_dir * 1e8) / 1e8;
            
            let g1 = F::infinity();
            for j in 0..n_features {
                let tmp = (largest_cov - cov[j]) / (AA - corr_eq_dir + F::epsilon());
                let g1 = if g1 > tmp && tmp >= 0 { tmp } else { g1 };  
            }

            let g2 = F::infinity();
            for j in 0..n_features {
                let tmp = (largest_cov + cov[j]) / (AA + corr_eq_dir + F::epsilon());
                let g2 = if g2 > tmp && tmp >= 0 { tmp } else { g2 };
            }
            let gamma_ = F::min(g1, F::min(g2, largest_cov / AA));

            drop = false;
            // TODO!!!
            // z = -coef[active] / (least_squares + tiny32)
            // z_pos = arrayfuncs.min_pos(z)
            if z_pos < gamma_ {
                // some coefficients have changed sign
            }
        }
    }
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
