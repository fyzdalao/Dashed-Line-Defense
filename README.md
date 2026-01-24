# Dashed Line Defense (DLD) - Implementation

This is the implementation of the paper "Dashed Line Defense: Plug-And-Play Defense Against Adaptive Score-Based Query Attacks" (AISTATS 2026).

### Citation

If you use this code or find our research helpful, please cite our paper.
> **Note:** The paper has been accepted by AISTATS 2026. This is a preliminary citation format. We will update the full BibTeX once the paper is officially available.

```bibtex
@inproceedings{fu2026dashed,
  title={Dashed Line Defense: Plug-And-Play Defense Against Adaptive Score-Based Query Attacks},
  author={Fu, Yanzhang and Guo, Zizheng and Luo, Jizhou},
  booktitle={29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026},
  note={To appear}
}
```

### Requirements

+ PyTorch 2.2
+ torchvision 0.17
+ CUDA 12.4
+ numpy

### Dataset Setup

This implementation requires the ImageNet validation set. Please:

Download the ImageNet validation set (50,000 images) from image-net.org.
Place the ImageNet validation images in the following structure:

<pre>
data/
└── ILSVRC2012_img_val/
    ├── ILSVRC2012_val_00000001.JPEG
    ├── ...
</pre>

### Usage

#### To evaluate DLD with default parameters:

<pre>
python main.py --defense=DLD --attack=square --tactic=none
</pre>

#### Main Available Options:

+ Model selection: --arch=wide_resnet50_2/regnet_y_1_6gf/maxvit_t
+ Defense method: --defense=None/DLD/AAASine/AAALinear/inRND
+ Attack tactics: --tactic=none/reverse/explore/sa
+ Defense parameters: --rnd_nu, --tau, --high_ratio, --s_step
+ Attack parameters: --eps, --budget

### License

The code in this repository is released under the BSD 3-Clause License.
Some components are re-implemented based on the AAA method,
following the original authors' implementation logic, and are included
for research and reproducibility purposes only.




