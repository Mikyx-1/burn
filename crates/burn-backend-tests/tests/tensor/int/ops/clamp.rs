use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_clamp_ops_int() {
    let tensor = TestTensorInt::<2>::from([[-3, 0, 2], [4, 7, 9]]);

    let output = tensor.clamp(0, 5);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 2], [4, 5, 5]]), false);
}

#[test]
fn should_support_clamp_min_max_ops_int() {
    let tensor = TestTensorInt::<2>::from([[-3, 0, 2], [4, 7, 9]]);

    tensor
        .clone()
        .clamp_min(0)
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 2], [4, 7, 9]]), false);

    tensor
        .clamp_max(5)
        .into_data()
        .assert_eq(&TensorData::from([[-3, 0, 2], [4, 5, 5]]), false);
}

#[test]
fn should_support_equal_clamp_bounds_for_signed_and_unsigned_ints() {
    TestTensorInt::<1>::from([-1, 0, 1])
        .clamp(0, 0)
        .into_data()
        .assert_eq(&TensorData::from([0, 0, 0]), false);

    TestTensorInt::<1>::from_data(TensorData::from([0u32, 1, 2]), &Default::default())
        .clamp(1u32, 1u32)
        .into_data()
        .assert_eq(&TensorData::from([1u32, 1, 1]), false);
}

#[test]
#[should_panic(expected = "clamp requires min to be less than or equal to max")]
fn should_reject_reversed_signed_int_bounds() {
    let tensor = TestTensorInt::<1>::from([-1, 0, 1]);

    let _ = tensor.clamp(2, -2);
}

#[test]
#[should_panic(expected = "clamp requires min to be less than or equal to max")]
fn should_reject_reversed_unsigned_int_bounds() {
    let tensor = TestTensorInt::<1>::from_data(TensorData::from([0u32, 1, 2]), &Default::default());

    let _ = tensor.clamp(2u32, 1u32);
}
