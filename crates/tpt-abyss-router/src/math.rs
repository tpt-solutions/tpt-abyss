//! Hand-rolled, dependency-light math used by the router.
//!
//! Kept simple and fully in `f32` with explicit bounds checking so the
//! router stays panic-free and tiny. A future learned MLP can reuse these
//! primitives.

use tpt_abyss_types::AbyssResult;

/// Matrix-vector product `y = W x + b` for `W` row-major of shape `[out, in]`.
///
/// # Errors
/// Returns [`tpt_abyss_types::AbyssError::Router`] on dimension mismatch.
pub fn matvec(w: &[f32], x: &[f32], b: &[f32]) -> AbyssResult<Vec<f32>> {
    if w.len() != x.len() * b.len() {
        return Err(tpt_abyss_types::AbyssError::Router(
            "matvec dimension mismatch".into(),
        ));
    }
    let mut y = vec![0.0f32; b.len()];
    for (o, (row, bo)) in y.iter_mut().zip(w.chunks_exact(x.len()).zip(b.iter())) {
        let mut acc = *bo;
        for (wi, xi) in row.iter().zip(x.iter()) {
            acc += wi * xi;
        }
        *o = acc;
    }
    Ok(y)
}

/// Numerically stable softmax over a mutable slice, in place.
#[inline]
pub fn softmax(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Returns the index of the maximum element, or `None` if empty.
#[inline]
#[allow(dead_code)]
pub fn argmax(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .fold(None, |best, (i, &val)| match best {
            None => Some((i, val)),
            Some((_bi, bv)) if val > bv => Some((i, val)),
            other => other,
        })
        .map(|(i, _)| i)
}

/// Clip `x` to `[lo, hi]`.
#[inline]
#[allow(dead_code)]
pub fn clip(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}
