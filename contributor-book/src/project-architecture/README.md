# Project Architecture

This section documents most major architectural decisions with the reasoning behind them.

**Sections**

- [Module](./module.md)
  - [Optimization](./module.md#optimization)
    - [Constraints](./module.md#constraints)
    - [Solution](./module.md#solution)
- [Serialization](./serialization.md)
  - [Constraints](./serialization.md#constraints)
  - [The burnpack format](./serialization.md#the-burnpack-format)
  - [The three record types](./serialization.md#the-three-record-types)
  - [Checkpointing in burn-train](./serialization.md#checkpointing-in-burn-train)
  - [Notes](./serialization.md#notes)
- [Tensor](./tensor.md)
- [Backend](./backend.md)
  - [Autodiff](./backend.md#autodiff)
