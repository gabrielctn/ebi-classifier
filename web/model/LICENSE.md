# Licence of the EBI Classifier model weights

The trained model (`ebi.onnx`, and the equivalent `final.safetensors` in the
desktop packages) is licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

It was obtained by fine-tuning the pretrained
[`convnextv2_tiny.fcmae_ft_in22k_in1k`](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)
weights (ConvNeXt V2, Woo et al., 2023, distributed through `timm`), which are
themselves released under CC BY-NC 4.0. The fine-tuned weights keep that
licence: you may share and adapt them for non-commercial purposes, provided
you give appropriate credit to both the original ConvNeXt V2 authors and this
project.

`ebi.onnx` is not stored in the git repository because of its size (112 MB).
Download it from the project's [release page](https://github.com/gabrielctn/ebi-classifier/releases) or its [Zenodo record](https://doi.org/10.5281/zenodo.24324392) and place it
in this folder. Its SHA-256 checksum is recorded in `config.json`, and the web
page verifies it before use.

The source code of the project is licensed separately, under the MIT licence
(see `LICENSE` at the root of the repository).
