# pytorch-generative-model-collections
Original : [[Tensorflow version]](https://github.com/hwalsuklee/tensorflow-generative-model-collections)

Pytorch implementation of various GANs.

This repository was re-implemented with reference to [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections) by [Hwalsuk Lee](https://github.com/hwalsuklee)

I tried to implement this repository as much as possible with [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections), But some models are a little different.

This repository is included code for CPU mode Pytorch, but i did not test. I tested only in GPU mode Pytorch.

## Dataset

- MNIST
- Fashion-MNIST
- CIFAR10
- SVHN
- STL10
- LSUN-bed
#### I only tested the code on MNIST and Fashion-MNIST.

## Generative Adversarial Networks (GANs)
### Lists (Table is borrowed from [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections))

*Name* | *Paper Link* | *Value Function*
:---: | :---: | :--- |
**GAN** | [Arxiv](https://arxiv.org/abs/1406.2661) | <img src = 'assets/equations/GAN.png' height = '70px'>
**LSGAN**| [Arxiv](https://arxiv.org/abs/1611.04076) | <img src = 'assets/equations/LSGAN.png' height = '70px'>
**WGAN**| [Arxiv](https://arxiv.org/abs/1701.07875) | <img src = 'assets/equations/WGAN.png' height = '105px'>
**WGAN_GP**| [Arxiv](https://arxiv.org/abs/1704.00028) | <img src = 'assets/equations/WGAN_GP.png' height = '70px'>
**DRAGAN**| [Arxiv](https://arxiv.org/abs/1705.07215) | <img src = 'assets/equations/DRAGAN.png' height = '70px'>
**CGAN**| [Arxiv](https://arxiv.org/abs/1411.1784) | <img src = 'assets/equations/CGAN.png' height = '70px'>
**infoGAN**| [Arxiv](https://arxiv.org/abs/1606.03657) | <img src = 'assets/equations/infoGAN.png' height = '70px'>
**ACGAN**| [Arxiv](https://arxiv.org/abs/1610.09585) | <img src = 'assets/equations/ACGAN.png' height = '70px'>
**EBGAN**| [Arxiv](https://arxiv.org/abs/1609.03126) | <img src = 'assets/equations/EBGAN.png' height = '70px'>
**BEGAN**| [Arxiv](https://arxiv.org/abs/1703.10717) | <img src = 'assets/equations/BEGAN.png' height = '105px'>  

#### Variants of GAN structure (Figures are borrowed from [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections))
<img src = 'assets/etc/GAN_structure.png' height = '600px'>

### Results for mnist
Network architecture of generator and discriminator is the exaclty sames as in [infoGAN paper](https://arxiv.org/abs/1606.03657).  
For fair comparison of core ideas in all gan variants, all implementations for network architecture are kept same except EBGAN and BEGAN. Small modification is made for EBGAN/BEGAN, since those adopt auto-encoder strucutre for discriminator. But I tried to keep the capacity of discirminator.

The following results can be reproduced with command:  
```
python main.py --dataset mnist --gan_type <TYPE> --epoch 50 --batch_size 64
```

#### Fixed generation
All results are generated from the fixed noise vector.

*Name* | *Epoch 1* | *Epoch 25* | *Epoch 50* | *GIF*
:---: | :---: | :---: | :---: | :---: |
GAN | <img src = 'assets/mnist_results/GAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/GAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/GAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/GAN_generate_animation.gif' height = '200px'>
LSGAN | <img src = 'assets/mnist_results/LSGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/LSGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/LSGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/LSGAN_generate_animation.gif' height = '200px'>
WGAN | <img src = 'assets/mnist_results/WGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_generate_animation.gif' height = '200px'>
WGAN_GP | <img src = 'assets/mnist_results/WGAN_GP_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_GP_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_GP_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/WGAN_GP_generate_animation.gif' height = '200px'>
DRAGAN | <img src = 'assets/mnist_results/DRAGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/DRAGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/DRAGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/DRAGAN_generate_animation.gif' height = '200px'>
EBGAN | <img src = 'assets/mnist_results/EBGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/EBGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/EBGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/EBGAN_generate_animation.gif' height = '200px'>
BEGAN | <img src = 'assets/mnist_results/BEGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/BEGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/BEGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/BEGAN_generate_animation.gif' height = '200px'>

#### Conditional generation
Each row has the same noise vector and each column has the same label condition.

*Name* | *Epoch 1* | *Epoch 25* | *Epoch 50* | *GIF*
:---: | :---: | :---: | :---: | :---: |
CGAN | <img src = 'assets/mnist_results/CGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/CGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/CGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/CGAN_generate_animation.gif' height = '200px'>
ACGAN | <img src = 'assets/mnist_results/ACGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/ACGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/ACGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/ACGAN_generate_animation.gif' height = '200px'>
infoGAN | <img src = 'assets/mnist_results/infoGAN_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/infoGAN_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/infoGAN_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/infoGAN_generate_animation.gif' height = '200px'>

#### InfoGAN : Manipulating two continous codes
All results have the same noise vector and label condition, but have different continous vector.

*Name* | *Epoch 1* | *Epoch 25* | *Epoch 50* | *GIF*
:---: | :---: | :---: | :---: | :---: |
infoGAN | <img src = 'assets/mnist_results/infoGAN_cont_epoch001.png' height = '200px'> | <img src = 'assets/mnist_results/infoGAN_cont_epoch025.png' height = '200px'> | <img src = 'assets/mnist_results/infoGAN_cont_epoch050.png' height = '200px'> | <img src = 'assets/mnist_results/infoGAN_cont_generate_animation.gif' height = '200px'>

#### Loss plot

*Name* | *Loss*
:---: | :---: |
GAN | <img src = 'assets/mnist_results/GAN_loss.png' height = '230px'>
LSGAN | <img src = 'assets/mnist_results/LSGAN_loss.png' height = '230px'>
WGAN | <img src = 'assets/mnist_results/WGAN_loss.png' height = '230px'>
WGAN_GP | <img src = 'assets/mnist_results/WGAN_GP_loss.png' height = '230px'>
DRAGAN | <img src = 'assets/mnist_results/DRAGAN_loss.png' height = '230px'>
EBGAN | <img src = 'assets/mnist_results/EBGAN_loss.png' height = '230px'>
BEGAN | <img src = 'assets/mnist_results/BEGAN_loss.png' height = '230px'>
CGAN | <img src = 'assets/mnist_results/CGAN_loss.png' height = '230px'>
ACGAN | <img src = 'assets/mnist_results/ACGAN_loss.png' height = '230px'>
infoGAN | <img src = 'assets/mnist_results/infoGAN_loss.png' height = '230px'>

### Results for fashion-mnist
Comments on network architecture in mnist are also applied to here.  
[Fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) is a recently proposed dataset consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

The following results can be reproduced with command:  
```
python main.py --dataset fashion-mnist --gan_type <TYPE> --epoch 50 --batch_size 64
```

#### Fixed generation
All results are generated from the fixed noise vector.

*Name* | *Epoch 1* | *Epoch 25* | *Epoch 50* | *GIF*
:---: | :---: | :---: | :---: | :---: |
GAN | <img src = 'assets/fashion_mnist_results/GAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/GAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/GAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/GAN_generate_animation.gif' height = '200px'>
LSGAN | <img src = 'assets/fashion_mnist_results/LSGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/LSGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/LSGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/LSGAN_generate_animation.gif' height = '200px'>
WGAN | <img src = 'assets/fashion_mnist_results/WGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/WGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/WGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/WGAN_generate_animation.gif' height = '200px'>
WGAN_GP | <img src = 'assets/fashion_mnist_results/WGAN_GP_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/WGAN_GP_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/WGAN_GP_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/WGAN_GP_generate_animation.gif' height = '200px'>
DRAGAN | <img src = 'assets/fashion_mnist_results/DRAGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/DRAGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/DRAGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/DRAGAN_generate_animation.gif' height = '200px'>
EBGAN | <img src = 'assets/fashion_mnist_results/EBGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/EBGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/EBGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/EBGAN_generate_animation.gif' height = '200px'>
BEGAN | <img src = 'assets/fashion_mnist_results/BEGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/BEGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/BEGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/BEGAN_generate_animation.gif' height = '200px'>

#### Conditional generation
Each row has the same noise vector and each column has the same label condition.

*Name* | *Epoch 1* | *Epoch 25* | *Epoch 50* | *GIF*
:---: | :---: | :---: | :---: | :---: |
CGAN | <img src = 'assets/fashion_mnist_results/CGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/CGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/CGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/CGAN_generate_animation.gif' height = '200px'>
ACGAN | <img src = 'assets/fashion_mnist_results/ACGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/ACGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/ACGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/ACGAN_generate_animation.gif' height = '200px'>
infoGAN | <img src = 'assets/fashion_mnist_results/infoGAN_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/infoGAN_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/infoGAN_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/infoGAN_generate_animation.gif' height = '200px'>

- ACGAN tends to fall into mode-collapse in [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections), but Pytorch ACGAN does not fall into mode-collapse.

#### InfoGAN : Manipulating two continous codes
All results have the same noise vector and label condition, but have different continous vector.

*Name* | *Epoch 1* | *Epoch 25* | *Epoch 50* | *GIF*
:---: | :---: | :---: | :---: | :---: |
infoGAN | <img src = 'assets/fashion_mnist_results/infoGAN_cont_epoch001.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/infoGAN_cont_epoch025.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/infoGAN_cont_epoch050.png' height = '200px'> | <img src = 'assets/fashion_mnist_results/infoGAN_cont_generate_animation.gif' height = '200px'>

#### Loss plot

*Name* | *Loss*
:---: | :---: |
GAN | <img src = 'assets/fashion_mnist_results/GAN_loss.png' height = '230px'>
LSGAN | <img src = 'assets/fashion_mnist_results/LSGAN_loss.png' height = '230px'>
WGAN | <img src = 'assets/fashion_mnist_results/WGAN_loss.png' height = '230px'>
WGAN_GP | <img src = 'assets/fashion_mnist_results/WGAN_GP_loss.png' height = '230px'>
DRAGAN | <img src = 'assets/fashion_mnist_results/DRAGAN_loss.png' height = '230px'>
EBGAN | <img src = 'assets/fashion_mnist_results/EBGAN_loss.png' height = '230px'>
BEGAN | <img src = 'assets/fashion_mnist_results/BEGAN_loss.png' height = '230px'>
CGAN | <img src = 'assets/fashion_mnist_results/CGAN_loss.png' height = '230px'>
ACGAN | <img src = 'assets/fashion_mnist_results/ACGAN_loss.png' height = '230px'>
infoGAN | <img src = 'assets/fashion_mnist_results/infoGAN_loss.png' height = '230px'>

## Folder structure
The following shows basic folder structure.
```
├── main.py # gateway
├── data
│   ├── mnist # mnist data (not included in this repo)
│   ├── ...
│   ├── ...
│   └── fashion-mnist # fashion-mnist data (not included in this repo)
│
├── GAN.py # vainilla GAN
├── utils.py # utils
├── dataloader.py # dataloader
├── models # model files to be saved here
└── results # generation results to be saved here
```

## Development Environment
* Ubuntu 16.04 LTS
* NVIDIA GTX 1080 ti
* cuda 9.0
* Python 3.5.2
* pytorch 0.4.0
* torchvision 0.2.1
* numpy 1.14.3
* matplotlib 2.2.2
* imageio 2.3.0
* scipy 1.1.0

## Acknowledgements
This implementation has been based on [tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections) and tested with Pytorch 0.4.0 on Ubuntu 16.04 using GPU.

