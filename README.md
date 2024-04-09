# Anatomically Conditioned Contrastive Unpaired Translation (ACCUT)

ACCUT is an extension of the "Contrastive Learning for Unpaired Image-to-Image Translation" by Park et al. It is an approach to retain semantic information when during style transfer. This method was tested only on optical coherence tomography data.

### Network architecture
The network architecture is the same as in the original CUT paper except that we added a segmentation decoder which passes features to the style decoder and thereby stresses semantic information. Standard crossentropy loss is used to train the segmentation pathway of the network. <br /> To separate shape from appearance, we update only encoder and style decoder when optimizing $`\mathcal{L}_{PatchNCE}`$ and $`\mathcal{L}_{GAN}`$. 

<p align="center">
<img src="./imgs/all_networks.png" alt="drawing" width="400"/>
</p>


### Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/2404.05409).
```
@misc{seibel2024anatomical,
      title={Anatomical Conditioning for Contrastive Unpaired Image-to-Image Translation of Optical Coherence Tomography Images}, 
      author={Marc S. Seibel and Hristina Uzunova and Timo Kepp and Heinz Handels},
      year={2024},
      eprint={2404.05409},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
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
