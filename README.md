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

----

| Model                    | Paper                            |Reconstruction | Samples |
|--------------------------|----------------------------------|---------------|---------|
|  VAE                     |https://arxiv.org/abs/1312.6114   |    ![][2]     | ![][1]  |
|  WAE - MMD (RBF Kernel)  |https://arxiv.org/abs/1711.01558  |               | ![][3]  |
|  WAE - MMD (IMQ Kernel)  |https://arxiv.org/abs/1711.01558  |               | ![][4]  |



### TODO
- [x] VanillaVAE
- [x] Conditional VAE
- [ ] Gamma VAE
- [x] Beta VAE
- [ ] DFC VAE
- [ ] InfoVAE (MMD-VAE)
- [x] WAE-MMD
- [ ] AAE
- [ ] TwoStageVAE
- [ ] VAE-GAN
- [x] HVAE (VAE with Vamp Prior)
- [ ] IWAE
- [ ] VLAE
- [ ] FactorVAE
- [ ] PixelVAE

[1]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/Vanilla%20VAE_25.png
[2]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_Vanilla%20VAE_25.png
[3]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/WAE_RBF_17.png
[4]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/WAE_IMQ_15.png




[python-image]: https://img.shields.io/badge/Python-3.5-ff69b4.svg
[python-url]: https://www.python.org/

[pytorch-image]: https://img.shields.io/badge/PyTorch-1.3-2BAF2B.svg
[pytorch-url]: https://pytorch.org/

[twitter-image]:https://img.shields.io/twitter/url/https/shields.io.svg?style=social
[twitter-url]:https://twitter.com/intent/tweet?text=Neural%20Blocks-Easy%20to%20use%20neural%20net%20blocks%20for%20fast%20prototyping.&url=https://github.com/AntixK/NeuralBlocks


[license-image]:https://img.shields.io/badge/license-Apache2.0-blue.svg
[license-url]:https://github.com/AntixK/PyTorch-VAE/blob/master/LICENSE.md
