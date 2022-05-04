# SeMask-FPN Inference

A gutted [SeMask](https://github.com/Picsart-AI-Research/SeMask-Segmentation) port with less 
strict dependencies that's set up to make performing inference and extracting pre-class features very easy. More [examples](https://github.com/GerardMaggiolino/SeMask-FPN-Inference#inference-and-features3) are given below, but after the quick installation, you can run the following to automatically download weights 
and extract pre-class features: 

```
import semask
model = semask.load_model("ade20k_base")
features = model.inference_features(batch, resize=True)
```

Please consider checking out or following the [original author's](https://github.com/praeclarumjj3) work. 

### Installation

Just install the very pared down dependency list (or use your existing environment that likely
has everything already) and install [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
As a lot of mmcv dependent training code / parallelism has been removed, most versions should work. 
```
python setup.py develop
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

For example, I tested the code with Python 3.9 and PyTorch 1.11.0+CUDA 10.2, and a prebuilt wheel 
was available with:
```
# cu_version = cu102
# torch_version = torch1.11.0
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html
```

If a wheel isn't available, compiling from source also works with: 
```
pip install mmcv-full
```

### Inference and Features

After running the setup.py, you should be able to just: 

```python
import semask

print(semask.models)                            # See all the available pretrained models 
model = semask.load_model("ade20k_base").eval()

with torch.no_grad():
    softmaxed_output = model.inference(batch)
    softmaxed_output_resized = model.inference(batch, rescale=True)
    print(softmaxed_output.shape)               # (Nx150xH/4xW/4) for ADE20K with 150 classes
    print(softmaxed_output_resized.shape)       # (Nx150xHxW) 
    
    features = model.inference_features(batch, rescale=True)
    print(features.shape)                       # (Nx256xHxW) 
```

The weights are downloaded and stored to `semask/weights`, pulled from the original repo.
The features are taken before the final convolution layer, after upsampling and aggregation. The loadable models 
are in `semask.models`, and are the pretrained weights provided by the original authors.
This includes `ade20k_<size>`, `cityscapes_<size>`, `coco10k_<size>` for sizes of `tiny, small, base, large`.

See the full [encoder_decode.py](https://github.com/GerardMaggiolino/SeMask-FPN-Inference/blob/main/semask/mmseg/models/segmentors/encoder_decoder.py) file for a full list of raw functions from the original SeMask implementation.

### Examples

Two simple inference files are included that generate the shown output given the following images. 

For clustering pre-class layer features, check out [examples/clustering.py](https://github.com/GerardMaggiolino/SeMask-FPN-Inference/blob/main/examples/clustering.py). Here's some example output from clustering the features obtained from: 
```
features = model.inference_features(img, rescale=True)
```
![](https://github.com/GerardMaggiolino/SeMask-FPN-Inference/blob/main/examples/terrain.png)
![](https://github.com/GerardMaggiolino/SeMask-FPN-Inference/blob/main/examples/clustered_image.png)

For running basic inference, check out [examples/inference.py](https://github.com/GerardMaggiolino/SeMask-FPN-Inference/blob/main/examples/inference.py). Here's some example output using the `cityscapes_base` model. 

![](https://github.com/GerardMaggiolino/SeMask-FPN-Inference/blob/main/examples/cityscapes.png)
![](https://github.com/GerardMaggiolino/SeMask-FPN-Inference/blob/main/examples/inference_image.png)

