# Einsum demo

Run from the workspace root:

```sh
cargo run -p burn-tensor --example einsum --features flex,autodiff
```

The demo exercises literal equations, a runtime equation with mixed operand
ranks, trace, MaskFormer-style mask prediction, and automatic differentiation.
Expected output:

```text
Matrix multiplication: [19.0, 22.0, 43.0, 50.0]
Runtime equation: [50.0, 110.0]
Trace: [5.0]
Mask prediction [1, 2, 1, 2]: [7.0, 10.0, 15.0, 22.0]
Gradient of dot(x, x): [2.0, 4.0, 6.0]
```

`einsum!` parses literal equations at compile time and generates checked,
left-to-right contraction stages. `Tensor::einsum` accepts runtime strings.
Both use the same parser and executor and support broadcasting, ellipses,
diagonals, and multiple operands. Scalar tensors use shape `[1]`.

The equation syntax follows the conventions documented by
[PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) and
[NumPy](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
Burn contracts operands from left to right and does not search for an optimized contraction
order. Scalar results use shape `[1]`; quantized operands are unsupported; and Float or Int
operands must share their dtype and device. See the
[implementation](../src/tensor/api/einsum/) and [tests](../tests/einsum_macros.rs) for details and
coverage.
