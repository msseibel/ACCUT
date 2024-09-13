# Anatomically Conditioned Contrastive Unpaired Translation (ACCUT)

ACCUT is an extension of the "Contrastive Learning for Unpaired Image-to-Image Translation" by Park et al. It is an approach to retain semantic information during style transfer. This method was tested only on optical coherence tomography data.

### Network architecture
The network architecture is the same as in the original CUT paper except that we added a segmentation decoder which passes features to the style decoder and thereby stresses semantic information. Standard crossentropy loss is used to train the segmentation pathway of the network. <br /> To separate shape from appearance, we update only encoder and style decoder when optimizing $`\mathcal{L}_{PatchNCE}`$ and $`\mathcal{L}_{GAN}`$. 

<p align="center">
<img src="./imgs/all_networks.png" alt="drawing" width="400"/>
</p>


### Dataset preparation
The dataset could be organized as follows:
```
dataset
├── DomainA
│   ├── subject1
│   │   ├── X-{eye}-{idx:03d}.tiff
│   │   ├── Y-{eye}-{idx:03d}.tiff
│   ├── subject2
...
├── DomainB
│   ├── subject1
│   │   ├── X-{eye}-{idx:03d}.tiff
│   │   ├── Y-{eye}-{idx:03d}.tiff (optional)
...
```

### Training:
To train the model using the source segmentation masks, you need to run the following command:
```bash
python train.py  --content_path PATH_TO_CONTENT --style_path PATH_TO_STYLE --name OUTDIR_EXPERIMENT_NAME --dataset_mode visotec_spectralis --model semcut --CUT_mode CUT --load_src_seg True
```

### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2404.05409).
```
@INPROCEEDINGS{10635513,
  author={Seibel, Marc S. and Uzunova, Hristina and Kepp, Timo and Handels, Heinz},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)}, 
  title={Anatomical Conditioning for Contrastive Unpaired Image-to-Image Translation of Optical Coherence Tomography Images}, 
  year={2024},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ISBI56570.2024.10635513}}
```

If you use the original [CUT](https://arxiv.org/pdf/2007.15651), [pix2pix](https://phillipi.github.io/pix2pix/) and [CycleGAN](https://junyanz.github.io/CycleGAN/) model included in this repo, please cite the following papers
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}


@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```


### Acknowledgments
 Our code is developed based on [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation). 
