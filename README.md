# Neural Art
Deep Learning for "Just as" painting styles
Inspired by Coursera - Deep Learning & Art : Neural Style Transfer

This tutorial can help you to use Neural Style transfer as described in <a href="https://arxiv.org/pdf/1508.06576v2.pdf">A Neural Algorithm of Artistic Style</a> by Leon A. Gatys, Alexander S. Ecker & Matthias Bethge.

It allows to create a new image (your picture) inspired by another such as a famous work of art.

We'll implement the neural style transfer algorithm by using the Tensroflow library.

# How to use
There are two images to send in the input of the neural network :
- the content image (1)
- the style image (2)

In the output we get the generated image merges the content image (1) with the style of (2).

The two images must have the same shapes but they could be modify, as other parameters, in **nst_utils.py** file :
```
class CONFIG:
    IMAGE_WIDTH = 400 # Width of the two input images.
    IMAGE_HEIGHT = 400 # Height of the two input images.
    COLOR_CHANNELS = 3 # RGB
    NOISE_RATIO = 0.6 # Ratio to use to generate a noisy image by adding random noise to the content image
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) # To substract the mean to match the expected input of VGG16
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat' # VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    STYLE_IMAGE = 'images/style.jpg' # Style image to use.
    CONTENT_IMAGE = 'images/content.jpg' # Content image to use.
    OUTPUT_DIR = 'output/' # Output directory
```
