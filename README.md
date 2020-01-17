<p align="center">
  <b>PyTorch VAE</b><br>
</p>
-----

A Collection of Variational AutoEncoders (VAEs) implemented in PyTorch.

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


| Model                    | Paper                            |Reconstruction | Samples |
|--------------------------|----------------------------------|---------------|---------|
|  VAE                     |https://arxiv.org/abs/1312.6114   |               | ![][1]  |
|  WAE - MMD (RBF Kernel)  |https://arxiv.org/abs/1711.01558  |               | ![][2]  |
|  WAE - MMD (IMQ Kernel)  |https://arxiv.org/abs/1711.01558  |               | ![][3]  |



### TODO
- [x] VanillaVAE
- [ ] Conditional VAE
- [ ] Gamma VAE
- [ ] Beta VAE
- [ ] DFC VAE
- [ ] InfoVAE (MMD-VAE)
- [x] WAE-MMD
- [ ] AAE
- [ ] TwoStageVAE
- [ ] VAE-GAN
- [ ] VAE with Vamp Prior
- [ ] IWAE
- [ ] VLAE
- [ ] FactorVAE
- [ ] PixelVAE

[1]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/Vanilla%20VAE_25.png
[2]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/WAE_RBF_17.png
[3]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/WAE_IMQ_15.png
