use burn_core as burn;

use burn::{
    Tensor,
    tensor::{Device, Gradients, container::TensorContainer},
};

use burn::module::{Module, ParamId};

use super::visitor::{GradientsParamsChangeDevice, GradientsParamsConverter};

/// Container of gradients keyed by parameter ID.
#[derive(Default, Debug)]
pub struct GradientsParams {
    container: TensorContainer<ParamId>,
}

impl GradientsParams {
    /// Create an empty [`GradientsParams`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract tensor gradients for the given [module](Module).
    ///
    /// This consumes the gradient container. See [`Self::from_module`] to extract gradients from a
    /// mutable container instead.
    pub fn from_grads<M: Module>(grads: Gradients, module: &M) -> Self {
        let mut grads = grads;
        Self::from_module(&mut grads, module)
    }

    /// Extract tensor gradients for the given [module](Module).
    pub fn from_module<M: Module>(grads: &mut Gradients, module: &M) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M>::new(grads, &mut grads_params, None);
        module.visit(&mut visitor);
        grads_params
    }

    /// Extract tensor gradients for the given [module](Module) and parameter IDs.
    pub fn from_params<M: Module>(grads: &mut Gradients, module: &M, params: &[ParamId]) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor =
            GradientsParamsConverter::<M>::new(grads, &mut grads_params, Some(params.to_vec()));
        module.visit(&mut visitor);
        grads_params
    }

    /// Get the gradient for the given [parameter ID](ParamId).
    ///
    /// # Notes
    ///
    /// Use [`Self::remove`] to retrieve a gradient only once.
    pub fn get<const D: usize>(&self, id: ParamId) -> Option<Tensor<D>> {
        self.container.get(&id)
    }

    /// Remove the gradient for the given [parameter ID](ParamId).
    pub fn remove<const D: usize>(&mut self, id: ParamId) -> Option<Tensor<D>> {
        self.container.remove(&id)
    }

    /// Register a gradient tensor for the given [parameter ID](ParamId).
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given [parameter ID](ParamId), it will be replaced.
    pub fn register<const D: usize>(&mut self, id: ParamId, value: Tensor<D>) {
        // TODO: always call value.inner() to make sure?
        self.container.register(id, value)
    }

    /// The number of registered gradient tensors.
    pub fn len(&self) -> usize {
        self.container.len()
    }

    /// Whether no gradient tensors are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Move each gradient tensor registered for the given [module](Module) to the specified device.
    pub fn to_device<M: Module>(mut self, device: &Device, module: &M) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{Module, ParamGroup};
    use burn::tensor::Distribution;
    use burn_nn::{Linear, LinearConfig};

    #[test]
    fn test_convert_grads() {
        let device = Device::default().autodiff();
        let layer_1 = layer(&device);
        let mut layer_2 = layer_1.clone();
        layer_2 = layer_2.fork(&device);
        let loss_1 = layer_1.forward(random_tensor(&device));
        let loss_2 = layer_2.forward(random_tensor(&device));
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer_1);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer_2);

        let group = ParamGroup::ids_from_module(layer_1.clone());
        let bias_2 = layer_2.bias.as_ref().unwrap();

        assert!(group.matches(&layer_2.weight.id, None));
        assert!(group.matches(&bias_2.id, None));
        assert_eq!(grads_1.len(), 2);
        assert_eq!(grads_2.len(), 2);
    }

    fn layer(device: &Device) -> Linear {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor(device: &Device) -> Tensor<2> {
        Tensor::<2>::random([2, 20], Distribution::Default, device)
    }
}
