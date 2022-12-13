# CSE 599G1 Final Project: Landscape Generation using Deep Convolutional Generative Adversarial Networks

- "[Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)" dataset by arnaud58 on Kaggle. This a dataset of 4,319 landscape images of varying sizes.

The directory structure for our project

```
├───data_preprocessed/preprocessed_64 // preprocessed 64x64 images
... // preprocessed 128x128 images are not uploaded but can be generated with preprocessing.py
├───data_preprocessed_256/preprocessed_256 // preprocessed 256x256 images
├───outputs // labeled below
└───src
│   |───DCGAN.py // 64x64 model
│   |───DCGAN128.py // 128x128 model
│   |───DCGAN256.py // 256x256 model
│   |───preprocessing.py // preprocessing code parameterized by desired resulting image size
│   |───train.py // 64x64 training code
│   |───train128.py // 128x128 training code
│   |───train256.py // 256x256 training code
```

## How to execute
- Use `preprocessing.py` to generate preprocessed dataset of desired size, saving to a directory of the form `data_preprocessed[SIZE]/preprocessed_[SIZE]`
- Run `train[SIZE].py`

## Example results
- Figure_1.png // DCGAN loss (initial experiment)
- Figure_2.png // 64x64 generated images (initial experiment)
- Figure_3.png // Additional 64x64 generated images (initial experiment)
- animation128.gif // DCGAN128
- animation64.gif // DCGAN64
- dcgan.png // DCGAN architecture
- image128.png // 128x128 generated images
- image64.png // 64x64 generated images
- loss128.png // DCGAN128 loss
- loss64.png // DCGAN loss
- output.mp4 // 64x64 generated images video
- stylegan.gif // HD StyleGAN gif (not used in our comparison)
- stylegan3.gif // 256x256 StyleGAN gif

64x64 generated images
![image64](https://user-images.githubusercontent.com/56491725/207198239-b3821b63-f9fb-4bad-917c-220d2215d978.png)

128x128 generated images
![image128](https://user-images.githubusercontent.com/56491725/207198248-40f60d55-3f2b-40c7-ab23-ffcc222822a0.png)



## References
[1] Alec Radford, Luke Metz, Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR (Poster) 2016.

[2] Ali Razavi, Aäron van den Oord, and Oriol Vinyals. 2019. Generating diverse high-fidelity images with VQ-VAE-2. Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, Article 1331, 14866–14876.

[3] Mahesh Gorijala and Ambedkar Dukkipati. “Image Generation and Editing with Variational Info Generative AdversarialNetworks.” ArXiv abs/1701.04568 (2017).

[4] T. Karras, S. Laine and T. Aila, "A Style-Based Generator Architecture for Generative Adversarial Networks" in IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 43, no. 12, pp. 4217-4228, 2021. doi: 10.1109/TPAMI.2020.2970919.
