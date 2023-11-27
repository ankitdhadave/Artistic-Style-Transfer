# Artistic-Style-Transfer
Assignment task for Dashtoon campus placement for the post of Research Engineer, Generative AI to generate using style transfer algorithm .  
#Artistic Style Transfer
# Project Overview
This project focuses on creating a deep learning model capable of adapting an existing work to resemble the aesthetic of any art. The model analyzes the artistic style of a selected artwork and applies similar stylistic features to a new, original artwork, creating a piece that seems as though it could have been created by the artist themselves.

Model Architecture
# Neural Network Design:
Utilizes the VGG19 model for feature extraction.
Implements a style transfer algorithm leveraging learned features to stylize input images.
python

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

Check if a GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found. Make sure TensorFlow is configured to use GPU.")

Load a pre-trained VGG19 model
base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

style_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block3_conv1', 'block3_conv3']
content_layer = 'block1_conv1'

style_outputs = [base_model.get_layer(name).output for name in style_layers]
content_output = base_model.get_layer(content_layer).output

model = models.Model(inputs=base_model.input, outputs=style_outputs + [content_output])

# Training
Dataset:
The model doesn't require traditional training as it's focused on style transfer rather than a classification task.
Style Transfer Implementation
Code for Style Transfer:

#Defines a function apply_style_transfer that takes content and style images and performs style transfer using VGG19 layers.
Uses pre-trained VGG19 weights for feature extraction.
python
Define a function to calculate the content loss
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

#Define a function to calculate the Gram matrix (used for style loss)
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

#Define a function to calculate the style loss
def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

#Define a function to apply style transfer
def apply_style_transfer(content_image, style_image, num_iterations=200, content_weight=1e6, style_weight=1e-7):
    content_image = tf.keras.preprocessing.image.img_to_array(content_image)
    content_image = tf.image.convert_image_dtype(content_image, dtype=tf.uint8)

    style_image = tf.keras.preprocessing.image.img_to_array(style_image)
    style_image = tf.image.convert_image_dtype(style_image, dtype=tf.uint8)

    content_image = tf.image.resize(content_image, (256, 256))
    style_image = tf.image.resize(style_image, (256, 256))

    content_image = tf.expand_dims(content_image, 0)
    style_image = tf.expand_dims(style_image, 0)

    content_image = tf.keras.applications.vgg19.preprocess_input(content_image)
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image)

    generated_image = tf.Variable(content_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    style_outputs = model(style_image)
    style_features = [style_layer[0] for style_layer in style_outputs[:-1]]
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            model_outputs = model(generated_image)
            style_features = model_outputs[:-1]
            content_feature = model_outputs[-1]

            content_loss = get_content_loss(content_feature, model(content_image)[-1])

            style_loss = 0
            for j in range(len(style_layers)):
                style_loss += get_style_loss(style_features[j][0], gram_style_features[j])

            total_loss = content_weight * content_loss + style_weight * style_loss

        grads = tape.gradient(total_loss, generated_image)
        opt.apply_gradients([(grads, generated_image)])

        if i % 100 == 0:
            print(f"Iteration {i}, Total loss: {total_loss.numpy()}")

    return generated_image.numpy()
Example usage:

python
Copy code
content_path = '/content/iit_kgp.jpg'
style_path = '/content/Vincent-van-Gogh-starry-Night.jpg'
content_img = image.load_img(content_path)
style_img = image.load_img(style_path)

result = apply_style_transfer(content_img, style_img)
content image = ![iit_kgp](https://github.com/ankitdhadave/Artistic-Style-Transfer/assets/127585274/8fa06d4a-4fb2-491b-831b-3727ae8f1dab)

styleimage = ![Vincent-van-Gogh-starry-Night](https://github.com/ankitdhadave/Artistic-Style-Transfer/assets/127585274/b1b44b50-f2f4-4c2e-b9b4-e42da8d29e45)

resuleted image= ![stylized_image202](https://github.com/ankitdhadave/Artistic-Style-Transfer/assets/127585274/8535ed7c-9b11-4ed9-87bd-2c9defa98e27)

How to Run the Notebook
Open the Jupyter Notebook(attached) using a Jupyter-compatible environment.
Execute each cell in the notebook sequentially.
Adjust parameters or configurations as needed in the style transfer function.
Provide paths to your content and style images in the example usage section.
Results
Example Stylized Image ![stylized_image202](https://github.com/ankitdhadave/Artistic-Style-Transfer/assets/127585274/5145bafc-1405-4fc8-9e08-9eb2c6e9557b)


Key Findings:

The VGG19 model successfully performs style transfer, adapting the content image to resemble the specified artistic style.
Acknowledgments
TensorFlow: https://www.tensorflow.org/
Matplotlib: https://matplotlib.org/

Note: Initially, I attempted to utilize a custom CNN architecture designed for my master's thesis on object detection. This architecture comprises five convolutional layers with max-pooling applied after each convolutional layer. However, I encountered an error while reading the weight file associated with this custom architecture. To proceed with and complete this assignment, I have opted to use a pre-trained model instead 
