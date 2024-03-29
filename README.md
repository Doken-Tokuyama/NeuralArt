# Neural Art
Deep Learning for "Just as" painting styles

Inspired by Coursera - Deep Learning & Art : Neural Style Transfer

This tutorial can help you to use Neural Style transfer as described in [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf) by Leon A. Gatys, Alexander S. Ecker & Matthias Bethge.

It allows to create a new image (your picture) inspired by another such as a famous work of art.

We'll implement the neural style transfer algorithm by using the Tensroflow library.

# How to use
## Parameters
There are two images to send in the input of the neural network :
- the content image (1)

![Content image](images/Content.jpg?raw=true "Content image")

- the style image (2)

![Style image](images/Style.jpg?raw=true "Style image")
> The Bridges of Amsterdam - [Leonid Afremov](https://afremov.com/)

In the output we get the generated image merging the content image (1) with the style of (2) :

![Result image](output/Result.png?raw=true "Result image")

The two images must have the same shape but this one could be modify, as other parameters, in the [nst_utils.py](nst_utils.py) file :
```
class CONFIG:
    IMAGE_WIDTH = 774 # You can modify the width of the two input images
    IMAGE_HEIGHT = 1145 # You can modify the height of the two input images
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat'  # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    STYLE_IMAGE = 'images/Style.jpg'  # Style image to use.
    CONTENT_IMAGE = 'images/Content.jpg'  # Content image to use.
    OUTPUT_DIR = 'output/' # Output directory
```

## VGG 19-layer
Note that in order to use this algorithm, you must download the pretrained model : [VGG 19-layer](https://www.kaggle.com/teksab/imagenetvggverydeep19mat#imagenet-vgg-verydeep-19.mat).

This technique is called the **transfer learning**. This model was trained on ImageNet and allow us to extract the feature maps to describe the content of our images.

Put the file in a new folder : **pretrained-model**.

## Dependencies
You'll need the following packages to run successfully the [main.py](main.py) file :
- [TensorFlow](https://www.tensorflow.org/install)
- [NumPy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
- [SciPy](https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt)
- [Pillow](https://pillow.readthedocs.io/en/3.3.x/installation.html#installation)

## License
GNU General Public License v3.0
