# SeMask-FPN Inference

A gutted [SeMask](https://github.com/Picsart-AI-Research/SeMask-Segmentation) port with less 
strict dependencies that's set up to make performing inference and extracting pre-class features very easy.

More examples are given below, but after the quick installation, you can run the following to automatically download weights 
and extract pre-class features: 
```
import semask
model = semask.load_model("ade20k_base")
features = model.inference_features(batch, resize=True)
```

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

model = semask.load_model("ade20k_base")
model.eval()
model.cuda()
with torch.no_grad():
    softmaxed_output = model.inference(batch.cuda())
    softmaxed_output_resized = model.inference(batch, rescale=True)
    print(softmaxed_output.shape)               # (Nx150xH/4xW/4) for ADE20K with 150 classes
    print(softmaxed_output_resized.shape)       # (Nx150xHxW) 
    features = model.inference_features(batch, rescale=True)
    print(features.shape)                       # (Nx256xH/4xW/4) 
```

The weights are downloaded and stored to `semask/weights`, pulled from the original repo.
The features are taken before the final convolution layer, after upsampling and aggregation. The loadable models 
are in semask.models, and are the pretrained weights provided by the original authors.
This includes `ade20k_<size>`, `cityscapes_<size>`, `coco10k_<size>` for sizes of `tiny, small, base, large`.

See `semask/mmseg/models/segmentors/encoder_decoder.py` for a full list of functions from the 
original SeMask implementation.

### Examples
