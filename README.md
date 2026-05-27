# Neural Style Transfer

Reimagining photos as paintings using deep learning.

Given a **content image** and a **style image**, this project generates a new image that preserves the structure of the content while adopting the artistic style of the painting — using a pretrained VGG19 network as a perceptual feature extractor.

---

## How it works

Based on the paper [**A Neural Algorithm of Artistic Style**](https://arxiv.org/abs/1508.06576) by Gatys et al. (2015).

The key idea:
- VGG19 is used as a fixed feature extractor — its weights are never updated
- A **content loss** (MSE between feature maps at `conv_4`) preserves the structure of the content image
- A **style loss** (MSE between Gram matrices at `conv_1` through `conv_5`) captures the artistic style
- A **total variation loss** encourages spatial smoothness and reduces noise
- The **input image** itself is the only learnable parameter — optimized using Adam


## Project structure

```text
Neural-Style-Transfer/
├── src/
│   ├── utils.py       # image loading, preprocessing, display, saving
│   ├── losses.py      # ContentLoss, StyleLoss, TotalVariationLoss, gram_matrix
│   ├── model.py       # VGG19 feature extractor and model builder
│   └── transfer.py    # core optimization loop
├── main.py            # entry point with argparse CLI
├── config.py          # all hyperparameters in one place
├── images/
│   ├── content/       # put your content images here
│   └── style/         # put your style images here
├── outputs/           # generated images saved here
└── requirements.txt

```

## Installation

```bash
git clone https://github.com/AyushNayal/Neural_Style_Transfer.git
cd Neural_Style_Transfer
pip install -r requirements.txt
```

---

## Usage

Basic usage:
```bash
python main.py --content images/content/photo.jpg --style images/style/painting.jpg
```

With custom settings:
```bash
python main.py \
  --content images/content/photo.jpg \
  --style images/style/painting.jpg \
  --steps 3000 \
  --style-weight 1e7 \
  --content-weight 2.0 \
  --output outputs/my_result.jpg
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--content` | required | Path to content image |
| `--style` | required | Path to style image |
| `--output` | `outputs/result.jpg` | Path to save result |
| `--steps` | 2000 | Number of optimization steps |
| `--style-weight` | 1e6 | Style loss weight |
| `--content-weight` | 2.0 | Content loss weight |
| `--tv-weight` | 1e-5 | Total variation loss weight |
| `--max-size` | 512 | Max image dimension in pixels |
| `--no-intermediates` | False | Skip intermediate output display |

---

## Technical details

### Why Gram matrices for style?
A Gram matrix captures correlations between feature maps at a given layer. These correlations encode texture and style information independent of spatial location — which is exactly what we want when transferring artistic style.

### Why VGG19?
VGG19's deep stack of conv layers produces rich hierarchical features. Shallow layers capture low-level textures and colors; deeper layers capture higher-level structure. This makes it ideal as a perceptual loss network.

### Why Total Variation loss?
The original Gatys et al. paper does not include TV loss. It was added here to reduce high-frequency noise and artifacts that appear during optimization, producing smoother and more visually appealing results.

---

## Requirements

- Python 3.8+
- PyTorch 2.5+
- torchvision 0.20+
- GPU recommended (CUDA) — CPU works but is significantly slower

---

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- Simonyan, K., & Zisserman, A. (2014). [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
