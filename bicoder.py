from tensorflow.keras import layers
from abc import ABC, abstractmethod


class BiCoder(layers.Layer, ABC):
    """
    BiCoder
    --------------------------
    Base superclass for both Encoder and Decoder in this Variational Autoencoder (VAE) project.

    This class defines a common interface shared by encoders and decoders. By inheriting from
    `tf.keras.layers.Layer`, it ensures that all subclasses integrate correctly with TensorFlow,
    while the abstract `call` method enforces a required forward-pass interface.

    This class should not be used directly. Instead, subclasses such as `Encoder` and `Decoder`
    extend `BiCoder` and override the `call` method with their specific implementations.

    Methods:
    1) __init__:
       Initializes the base BiCoder layer by calling the parent constructor from
       `layers.Layer`. This enables standard Keras behavior such as variable tracking
       and compatibility with `__call__`.

    2) call (abstract):
       Defines the required forward-pass interface that must be implemented by subclasses.
       This method enforces consistency across Encoder and Decoder:
       - Encoder: inputs = x (mini-batch of images) → returns (mu, std) for q(z|x)
       - Decoder: inputs = z (latent vectors)       → returns (mu, std) for p(x|z)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def call(self, inputs):
        """
        Forward pass interface to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must override BiCoder.call(x).")
