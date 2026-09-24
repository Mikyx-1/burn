use super::*;
use burn_tensor::{Shape, TensorData};

#[test]
fn test_cartesian_grid() {
    let device = Default::default();

    // Test a single element tensor
    let tensor: TestTensorInt<2> = TestTensorInt::<1>::cartesian_grid([1], &device);
    tensor
        .into_data()
        .assert_eq(&TensorData::from([[0]]), false);

    // Test for a 2x2 tensor
    let tensor: TestTensorInt<3> = TestTensorInt::<2>::cartesian_grid([2, 2], &device);
    tensor.into_data().assert_eq(
        &TensorData::from([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]),
        false,
    );
}

#[test]
#[should_panic(expected = "Cartesian grid shape rank must be 2, got 1")]
fn cartesian_grid_rejects_too_few_runtime_dimensions() {
    let device = Default::default();
    let _: TestTensorInt<3> = TestTensorInt::<2>::cartesian_grid(Shape::new([2]), &device);
}

#[test]
#[should_panic(expected = "Cartesian grid shape rank must be 2, got 3")]
fn cartesian_grid_rejects_too_many_runtime_dimensions() {
    let device = Default::default();
    let _: TestTensorInt<3> = TestTensorInt::<2>::cartesian_grid(Shape::new([2, 3, 4]), &device);
}
