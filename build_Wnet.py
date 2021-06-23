import ncut_loss
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
class Wnet(tf.keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        input_shape,
        

    ):
        super(Wnet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.neighbour_filter = ncut_loss.neighbor_filter(input_shape)

    def compile(
        self,
        optimizer,
        loss_fn_segmentation,
        loss_fn_reconstruction
    ):
        super(Wnet, self).compile()

        self.optimizer = optimizer
        self.loss_fn_segmentation = loss_fn_segmentation
        self.loss_fn_reconstruction = loss_fn_reconstruction #keras.losses.MeanSquaredError()
        
    def call(self, inputs, training=False):
      output = self.decoder(self.encoder(inputs))
      return output

    @tf.function
    def test_step(self,batch_data,reconstruction_loss_weight):
        image= batch_data

        result_encoder = self.encoder(image)
        result_decoder = self.decoder(result_encoder)
        loss_decoder = reconstruction_loss_weight*self.loss_fn_reconstruction(image,result_decoder)
        loss_encoder = self.loss_fn_segmentation(image,result_encoder,self.neighbour_filter)

        return {
            "loss_encoder": loss_encoder,
            "loss_decoder": loss_decoder,
        }

    @tf.function
    def train_step(self, batch_data,reconstruction_loss_weight,sigma=0.00001,blur_kernel=(10,10),noise_amp=0):
        
        image = batch_data
        image_blurred = tfa.image.gaussian_filter2d( image,(blur_kernel,blur_kernel),sigma)
        noise = np.random.normal(0, noise_amp, image.shape)
        image_blurred = image_blurred + noise
        image_blurred = tf.clip_by_value(image_blurred, clip_value_min=-1, clip_value_max=1)

        
        with tf.GradientTape() as tape:
          result_encoder = self.encoder(image_blurred)
          loss_encoder = self.loss_fn_segmentation(image,result_encoder,self.neighbour_filter)
          # REGULARISATION
          for layer in self.encoder.layers:
            loss_encoder+=tf.math.reduce_sum(layer.losses)
    
        grads_encoder_1 = tape.gradient(loss_encoder, self.encoder.trainable_variables)
          
        self.optimizer.apply_gradients(
            zip(grads_encoder_1, self.encoder.trainable_variables)
        )


        with tf.GradientTape() as tape:
          result_encoder = self.encoder(image_blurred)
          result_decoder = self.decoder(result_encoder)
          loss_decoder = reconstruction_loss_weight*self.loss_fn_reconstruction(image,result_decoder)
          # REGULARISATION
          for layer in self.encoder.layers+self.decoder.layers:
            loss_decoder+=tf.math.reduce_sum(layer.losses)      


        grads_2 = tape.gradient(loss_decoder, self.encoder.trainable_variables+self.decoder.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads_2, self.encoder.trainable_variables+self.decoder.trainable_variables)
        )
        
      

        return {
            "loss_encoder": loss_encoder,
            "loss_decoder": loss_decoder,
        }
