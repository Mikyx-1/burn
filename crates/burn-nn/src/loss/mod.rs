use burn_core::tensor::{Int, Tensor};

const BINARY_TARGET_ERROR: &str = "All target values must be either -1 or 1.";

pub(crate) fn assert_binary_float_targets<const D: usize>(targets: &Tensor<D>) {
    let valid = targets
        .clone()
        .equal_scalar(1.0)
        .bool_or(targets.clone().equal_scalar(-1.0))
        .all()
        .into_scalar::<bool>();
    assert!(valid, "{BINARY_TARGET_ERROR}");
}

pub(crate) fn assert_binary_int_targets<const D: usize>(targets: &Tensor<D, Int>) {
    let valid = targets
        .clone()
        .equal_scalar(1)
        .bool_or(targets.clone().equal_scalar(-1))
        .all()
        .into_scalar::<bool>();
    assert!(valid, "{BINARY_TARGET_ERROR}");
}

mod binary_cross_entropy;
mod cosine_embedding;
mod cross_entropy;
mod ctc;
mod gaussian_nll;
mod hinge_embedding;
mod huber;
mod kldiv;
mod lp_loss;
mod margin_ranking;
mod mse;
mod multi_margin;
mod poisson;
mod reduction;
mod rnnt;
mod smooth_l1;
mod soft_margin;
mod triplet_margin;

pub use binary_cross_entropy::*;
pub use cosine_embedding::*;
pub use cross_entropy::*;
pub use ctc::*;
pub use gaussian_nll::*;
pub use hinge_embedding::*;
pub use huber::*;
pub use kldiv::*;
pub use lp_loss::*;
pub use margin_ranking::*;
pub use mse::*;
pub use multi_margin::*;
pub use poisson::*;
pub use reduction::*;
pub use rnnt::*;
pub use smooth_l1::*;
pub use soft_margin::*;
pub use triplet_margin::*;
