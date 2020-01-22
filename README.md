<h1 align="center">
  <b>PyTorch VAE</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.5-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.3-2BAF2B.svg" /></a>
       <a href= "https://github.com/AntixK/PyTorch-VAE/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
         <a href= "https://twitter.com/intent/tweet?text=PyTorch-VAE:%20Collection%20of%20VAE%20models%20in%20PyTorch.&url=https://github.com/AntixK/PyTorch-VAE">
        <img src="https://img.shields.io/twitter/url/https/shields.io.svg?style=social" /></a>

</p>

A Collection of Variational AutoEncoders (VAEs) implemented in PyTorch with focus on reproducibility. The aim of this project is to provide
a quick and simple working example for many of the cool VAE models out there. All the models are trained on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
for consistency and comparison. The architecture of all the models are kept as similar as possible with the same layers, except for cases where the 
original paper demands a radically different architecture.

### Requirements
- Python >= 3.5
- PyTorch >= 1.3
- Pytorch Lightning >= 0.5.3 ([GitHub Repo](https://github.com/PyTorchLightning/pytorch-lightning/tree/deb1581e26b7547baf876b7a94361e60bb200d32))

### Installation
```
$ git clone https://github.com/AntixK/PyTorch-VAE
$ cd PyTorch-VAE
$ pip install -r requirements.txt
```

### Usage
```
$ cd PyTorch-VAE
$ python run.py -c configs/<config-file-name.yaml>
```
Config file template
```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: 3
  latent_dim: 

exp_params:
  data_path: "<path to the celebA dataset>"
  img_size: 64    # Models are designed to work for this size
  batch_size: 64  # Better to have a square number
  LR: 0.005

trainer_params:
  gpus: 1         
  max_nb_epochs: 50

logging_params:
  save_dir: "logs/"
  name: "<experiment name>"
  manual_seed: 
```


----

| Model                 | Paper                                            |Reconstruction | Samples |
|-----------------------|--------------------------------------------------|---------------|---------|
| VAE                   |[Link](https://arxiv.org/abs/1312.6114)           |    ![][2]     | ![][1]  |
| WAE - MMD (RBF Kernel)|[Link](https://arxiv.org/abs/1711.01558)          |    ![][4]     | ![][3]  |
| WAE - MMD (IMQ Kernel)|[Link](https://arxiv.org/abs/1711.01558)          |    ![][6]     | ![][5]  |
| Beta-VAE              |[Link](https://openreview.net/forum?id=Sy2fzU9gl) |    ![][8]     | ![][7]  |
| IWAE (5 Samples)      |[Link](https://arxiv.org/abs/1804.03599)          |    ![][10]    | ![][9]  |
| DFCVAE                |[Link](https://arxiv.org/abs/1610.00291)          |    ![][12]    | ![][11] |

<!--| Disentangled Beta-VAE |[Link](https://arxiv.org/abs/1804.03599)          |    ![][10]     | ![][9] |-->



### TODO
- [x] VanillaVAE
- [ ] Conditional VAE
- [ ] Gamma VAE
- [x] Beta VAE
- [ ] Beta TC-VAE
- [x] DFC VAE
- [ ] InfoVAE (MMD-VAE)
- [x] WAE-MMD
- [ ] AAE
- [ ] TwoStageVAE
- [ ] VAE-GAN
- [ ] Vamp VAE
- [ ] HVAE (VAE with Vamp Prior)
- [x] IWAE
- [ ] VLAE
- [ ] FactorVAE
- [ ] PixelVAE
- [ ] VQVAE
- [ ] StyleVAE

### Contributing
If you have trained a better model using these implementations by finetuning the hyper-params in the config file,
I would be happy to include your result (along with your config file) in this repo, citing your name &#1F607	.


[1]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/Vanilla%20VAE_25.png
[2]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_Vanilla%20VAE_25.png
[3]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/WAE_RBF_18.png
[4]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_WAE_RBF_19.png
[5]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/WAE_IMQ_15.png
[6]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_WAE_IMQ_15.png
[7]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/BetaVAE_B_20.png
[8]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_BetaVAE_B_20.png
[9]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/IWAE_19.png
[10]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_IWAE_19.png
[11]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/DFCVAE_49.png
[12]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_DFCVAE_40.png


[python-image]: https://img.shields.io/badge/Python-3.5-ff69b4.svg
[python-url]: https://www.python.org/

[pytorch-image]: https://img.shields.io/badge/PyTorch-1.3-2BAF2B.svg
[pytorch-url]: https://pytorch.org/

[twitter-image]:https://img.shields.io/twitter/url/https/shields.io.svg?style=social
[twitter-url]:https://twitter.com/intent/tweet?text=Neural%20Blocks-Easy%20to%20use%20neural%20net%20blocks%20for%20fast%20prototyping.&url=https://github.com/AntixK/NeuralBlocks


[license-image]:https://img.shields.io/badge/license-Apache2.0-blue.svg
[license-url]:https://github.com/AntixK/PyTorch-VAE/blob/master/LICENSE.md
