# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support override_scale in Resize


import numpy as np
import mmcv
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
import torch
from monai import transforms as monai_transforms


class MONAIWrapper(object):
    """Wrapper for MONAI transforms
    
    Example:
    transforms.RandShiftIntensity(-.5,prob=1),
    transforms.RandBiasField(prob=1,coeff_range=(0, 0.2)),
    transforms.RandScaleIntensity(-.2,prob=1),
    transforms.RandAdjustContrast(prob=1., gamma=(0.5, 4.5)),
    transforms.RandHistogramShift(num_control_points=10, prob=1.),
    transforms.RandGaussianSharpen(sigma1_x=(0.5, 1.0), sigma1_y=(0.5, 1.0), sigma1_z=(0.5, 1.0), sigma2_x=0.5, sigma2_y=0.5, sigma2_z=0.5, alpha=(10.0, 30.0), approx='erf', prob=1)
    MONAIWrapper('RandShiftIntensity', prob=1, offsets=-.5)
    """
    def __init__(self, transform_name, **kwargs):
        
        self.__name__ = transform_name
        self.transform = monai_transforms.__dict__[transform_name](**kwargs)
        
    def __call__(self, results):
        #if random.random() < self.prob:
        if self.__name__.endswith('d'):
            results['img'] = np.transpose(results['img'], (2,0,1))
            results['gt_semantic_seg'] = results['gt_semantic_seg'][None]
            results = self.transform(results)
            results['gt_semantic_seg'] = results['gt_semantic_seg'][0].numpy().astype(int)
            results['img'] = np.transpose(results['img'], (1,2, 0)) # transpose converts from monai tensor to numpy
            torch.cuda.empty_cache()
        else:
            img = results['img']
            img = self.transform(img).numpy()
            results['img'] = img
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(prob={0})'.format(self.prob)



class GINAugment(object):
    def __init__(self, prob, in_channels, out_channels, num_layers, interm_channels) -> None:
        self.prob = prob
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.transform = rand_conv.GINGroupConv(
            OUT_CHANNELS=out_channels, IN_CHANNELS=in_channels, INTERM_CHANNELS=interm_channels,N_LAYER=num_layers)

    def __call__(self, results):
        
        
        img = results['img'].copy()
        if len(img.shape)==2:
            img = img[:,:,None]
        if random.random() < self.prob:
            img = torch.Tensor(img)#.contiguous()
            img = img.permute(2,0,1)
            c = 0
            while len(img.shape)<4:
                img = img.unsqueeze(0)
                c+=1
            #print(img.shape)
            img = self.transform(img)
            for _ in range(c):
                img = img.squeeze(0)
            img = img.permute(1,2,0)
            results['img'] = img.numpy()
        else:
            if img.shape[2]!=self.out_channels:
                assert img.shape[2]==1
                img = np.repeat(img, 3, axis=2)
            results['img'] = img
            
        return results
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(in_channels={0}, out_channels={1}, num_layers={2}, prob={3})'.format(self.in_channels, self.out_channels, self.num_layers, self.prob)




import numpy as np
from scipy import signal, ndimage
from oct_utils import domain_adaptation
from skimage import exposure
# copy to oct_utils or even more general
def center_crop_bscan(x, size):
    H_in, W_in = x.shape
    half_H = int(np.floor(size[0]//2))
    half_W = int(np.floor(size[1]//2))
    center = (H_in//2, W_in//2)
    assert (center[0] - half_H >= 0) & (center[0] + half_H < H_in)
    assert (center[1] - half_W >= 0) & (center[1] + half_W < W_in)
    x = x[center[0]-half_H:center[0]-half_H + size[0], 
          center[1]-half_W:center[1]-half_W + size[1]]
    return x

def svdna(k, src_img, target_img, alpha=0.5):
    """
    Takes the noise from the `target_img` and applies it to `src_img`.
    
    :param k: Number of singular values describing the content of the image.
    :param target_img:
    :param src_img:
    :param alpha:
    :return:
    """
    assert len(src_img.shape)==2
    assert (np.array(src_img.shape)==np.array(target_img.shape)).all(),"Images must have same shape for content + style."
    
    try:
        u_target,s_target,vh_target=np.linalg.svd(target_img, full_matrices=False)
        u_source,s_source,vh_source=np.linalg.svd(src_img, full_matrices=False)
    
        if k=='auto':
            k = np.argwhere(np.cumsum(s_source/np.sum(s_source)) > .5)[0][0]
    
        if k>len(s_target):
            raise ValueError(f"k must be smaller than the number of singular values ({len(s_target)}) of the target image.")
        thresholded_singular_target=s_target
        thresholded_singular_target[0:k]=0

        thresholded_singular_source=s_source
        thresholded_singular_source[k:]=0

        target_style=np.array(
            [np.dot(u_target, np.dot(np.diag(thresholded_singular_target), vh_target))])
        content_src=np.array(
            [np.dot(u_source, np.dot(np.diag(thresholded_singular_source), vh_source))])
        
        #content_trgt     = target_img  - target_style
        noise_adapted_im = content_src + target_style

        noise_adapted_im_clipped=np.squeeze(noise_adapted_im)#.clip(0,255).astype(np.uint8)
    
    except np.linalg.LinAlgError:
        noise_adapted_im_clipped = src_img.copy()
        print("SVD did not converge. Only applying histogram matching.")
        
    svdna_im = exposure.match_histograms(noise_adapted_im_clipped, target_img)
    svdna_im = svdna_im*alpha + noise_adapted_im_clipped * (1-alpha)

    return svdna_im

import cv2

class SimpleIntensityMatching():
    def __init__(self):
        self.retina_classes = [1,2,3]
        self.gb_class = 0
        self.choroid_class = 4

    def __call__(self, results):
        pass

    def get_boundary(self, mask1, mask2):
        mask1 = mask1.astype(np.uint8)#.cpu().numpy()
        mask2 = mask2.astype(np.uint8)#.cpu().numpy()
        kernel = np.ones((5,5),np.uint8)
        mask1 = cv2.dilate(mask1, kernel, iterations=1)
        mask2 = cv2.dilate(mask2, kernel, iterations=1)
        boundary = (mask1 == 1) & (mask2 == 1)
        return boundary

    def transform_cpu(self, src_img, style_image, src_mask, style_mask):
        
        mask_retina = (src_mask==1) | (src_mask==2) | (src_mask==3)
        mask_gb = (src_mask==self.gb_class)
        mask_choroid = (src_mask==self.choroid_class)
        intensity_retina  = src_img[mask_retina] 
        intensity_gb      = src_img[mask_gb]
        intensity_choroid = src_img[mask_choroid]


        valid_mask_choroid = mask_choroid.sum() > 5
        valid_mask_retina = mask_retina.sum() > 5
        valid_mask_gb = mask_gb.sum() > 5

        #print('Avg intensities: ', src_img.mean(),intensity_retina.mean(), intensity_gb.mean(), intensity_choroid.mean())
        if valid_mask_retina:
            src_img[mask_retina] -= intensity_retina.mean()
            src_img[mask_retina] /= (src_img[mask_retina].std() + 1e-6)
        if valid_mask_gb:
            src_img[mask_gb] -= intensity_gb.mean()
            src_img[mask_gb] /= (src_img[mask_gb].std() + 1e-6)
        if valid_mask_choroid:
            src_img[mask_choroid] -= intensity_choroid.mean()
            src_img[mask_choroid] /= (src_img[mask_choroid].std() + 1e-6)
            src_img[mask_choroid]-=src_img[mask_choroid].min()

        #src_img, _ , _ = normalize(src_img)
        if valid_mask_choroid:
            src_img[mask_choroid]*=np.random.uniform(1.0,1.5)
        src_img, _ , _ = normalize(src_img)
        
        src_img_blurred = cv2.GaussianBlur(src_img.copy(),(9,9), sigmaX=0, sigmaY=0)
        
        if valid_mask_retina and valid_mask_gb:
            # get boundary between retina and choroid
            boundary_rc = self.get_boundary(mask_retina, mask_choroid)
            src_img[boundary_rc] = src_img_blurred[boundary_rc]

        if valid_mask_retina and valid_mask_gb:
            boundary_rgb = self.get_boundary(mask_retina, mask_gb)
            # gaussian blur on equal intensity image
            src_img[boundary_rgb] = src_img_blurred[boundary_rgb]
        return src_img
    

    def transform(self, src_img, style_img, src_mask, style_mask, output_format='N3HW'):
        if type(src_img)==torch.Tensor:
            # get device from tensor
            device = src_img.device
            backend_src = 'torch'    
    
        src_shape = src_img.shape
        if len(src_shape) == 4:
            transformed = np.zeros((src_shape[0],*src_shape[2:]))
            for i in range(src_shape[0]):
                transformed[i] = self.transform_cpu(src_img[i][0].cpu().numpy(), None, src_mask[i][0].cpu().numpy(), None)
            transformed = np.repeat(transformed[:,None],3, axis=1)
            assert output_format=='N3HW'

        if backend_src=='torch':
            transformed = torch.Tensor(transformed).to(device)
        return transformed




class SVDNA():
    def __init__(self, k_trunc, alpha, direction='src2tgt', prob=0.5, ksize=3):
        self.direction = direction
        self.prob = prob
        #self.trg = trg
        self.ksize = ksize
        if type(k_trunc)== int:
            self.k_trunc = (k_trunc, k_trunc)
        else:
            self.k_trunc = k_trunc
        if type(alpha)== float:
            self.alpha = (alpha, alpha)
        else:
            self.alpha = alpha
        self.min_patch_size = (150, 400) # this value is baded on the size of the visotec images
        
    def standardize_filter(self, ksize):
        def call(x):
            mean_kernel = np.ones((ksize, ksize))/ ksize**2
            local_mean = signal.convolve2d(x, mean_kernel, mode='same')
            return signal.convolve2d((x - local_mean) ** 2, mean_kernel, mode='same')
        return call
    
    def extract_largest(self, mask_valid):
        label_mask_valid, n_regions = ndimage.label(mask_valid)
        mask_size = {}
        for c in range(1, n_regions+1):
            mask_size[c] = np.sum(label_mask_valid==c)
        largest_component = max(mask_size, key=mask_size.get)
        return label_mask_valid == largest_component

    def get_bbox_from_mask(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 

    def stitchimages(self, images, minimum_target_size, min_patch_size=(160, 800)):
        images = [img for img in images if img.shape[0]>=min_patch_size[0] and img.shape[1]>=min_patch_size[1]]
        shapes = np.array([img.shape for img in images])
        h,w = np.min(shapes,axis=0)
        # crop all images to the smallest common size
        images = [img[:h, :w] for img in images]
        
        num_images_h = int(np.ceil(minimum_target_size[0]/h))
        num_images_w = int(np.ceil(minimum_target_size[1]/w))
        
        canvas = np.zeros((num_images_h * h, num_images_w * w))
        for idx_h in range(num_images_h):
            for idx_w in range(num_images_w):
                canvas[idx_h*h:(idx_h+1)*h, idx_w*w:(idx_w+1)*w] = images[int(idx_h*num_images_w)+idx_w]
        return canvas[:minimum_target_size[0], :minimum_target_size[1]]

    def get_valid_mask(self, std_img, border_size, v_mask=1):

        mask_ignore = np.zeros_like(std_img)
        mask_ignore[:border_size,:]  = v_mask
        mask_ignore[-border_size:,:] = v_mask
        mask_ignore[:,:border_size]  = v_mask
        mask_ignore[:,-border_size:] = v_mask
        mask_ignore[std_img<1e-8] = v_mask
        mask_valid = v_mask - mask_ignore
        mask_valid = ndimage.morphology.binary_fill_holes(mask_valid)
        mask_valid = self.extract_largest(mask_valid)
        return mask_valid

    def ignore_padded(self, style_images):
        # Spectralis images have some padded areas which we need to ignore for histogram matching
        for i in range(len(style_images)):
            img = style_images[i]
            o = self.standardize_filter(self.ksize)(img)
            mask_valid = self.get_valid_mask(o, border_size=self.ksize-1)
            bbox = self.get_bbox_from_mask(mask_valid)
            style_images[i] = img[bbox[0]:bbox[1], bbox[2]: bbox[3]]
        return style_images
    
    def transform_cpu(self, src_img, style_image):
        # get target image to same size as source image
        src_shape = src_img.shape
        style_shape = style_image.shape
        if src_shape[0] > style_shape[0] or src_shape[1] > style_shape[1]:
            #if len(style_image)==1:
            style_image = [style_image, 
                           style_image[:,::-1], 
                           style_image[::-1], 
                           style_image[::-1,::-1]]
            trg_img = self.stitchimages(style_image, minimum_target_size=src_shape, min_patch_size=self.min_patch_size)
        elif (src_shape[0] < style_shape[0]) and (src_shape[1] < style_shape[1]):
            trg_img = center_crop_bscan(style_image, src_shape)
        elif (src_shape[0] == style_shape[0]) and (src_shape[1] == style_shape[1]):
            trg_img = style_image
            
        k_trunc = np.random.randint(self.k_trunc[0], self.k_trunc[1])
        alpha = np.random.uniform(self.alpha[0], self.alpha[1])
        transformed = domain_adaptation.svdna(k_trunc, src_img, trg_img, alpha)

    def transform(self, src_img, style_image, output_format='HW'):
        backend_style = 'numpy'
        backend_src = 'numpy'
        if type(src_img)==torch.Tensor:
            # get device from tensor
            device = src_img.device
            src_img = src_img.cpu().numpy()
            backend_src = 'torch'    
            assert type(style_image)==torch.Tensor
            style_image = style_image.cpu().numpy()
            

        src_shape = src_img.shape
        style_shape = style_image.shape

        if len(src_shape)==4:
            transformed = np.zeros((src_shape[0],*src_shape[2:]))
            for i in range(src_shape[0]):
                transformed[i] = self.transform_cpu(src_img[i][0], style_image[i][0])
            transformed = np.repeat(transformed[:,None],3, axis=1)
            assert output_format=='N3HW'
        else:
            transformed = self.transform_cpu(src_img, style_image)

        if backend_src=='torch':
            transformed = torch.Tensor(transformed).to(device)
        
        if output_format=='N3HW' and len(src_shape)==2:
            if backend_src == 'torch':
                # repeat for 3 channels
                transformed = torch.unsqueeze(transformed, 0)
                transformed = torch.repeat_interleave(transformed, 3, dim=0)
            else:
                raise NotImplementedError
            
        return transformed
    
    def __call__(self, results):
        
        # Skipped for online augmentation -> due to high computational cost and it could average out.
        # Exclude non-valid areas from SVDNA
        #if self.trg in ['Spectralis','Topcon', 'Cirrus']:
        #    style_images = self.ignore_padded(style_images)
        #elif self.trg == 'Visotec':
        #    # do nothing
        #    pass
        if len(results['img'].shape)==2:
            results['img'] = results['img'][None]
        if len(results['target_img'].shape)==2:
            results['target_img'] = results['target_img'][None]
            
        assert len(results['img'].shape)==3
        assert len(results['target_img'].shape)==3
        assert results['img'].shape[0]==1
        assert results['target_img'].shape[0]==1

        if (random.random() < self.prob) and (self.direction in ['both', 'src2tgt']):
            style_image = results['target_img'].copy()[0]
            src_img = results['img'].copy()[0]
            src2tgt = self.transform(src_img, style_image)
        else:
            src2tgt = results['img'].copy()
        if (random.random() < self.prob) and (self.direction in ['both', 'tgt2src']):  
            style_image = results['img'].copy()
            src_img = results['target_img'].copy()
            tgt2src = self.transform(src_img, style_image)
        else:
            tgt2src = results['target_img'].copy()
        results['target_img'] = tgt2src    
        results['img'] = src2tgt
        return results
        



class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 override_scale=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.override_scale = override_scale

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results or self.override_scale:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str



class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction']).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'



class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            # Opencv throws an error, if one dim of the image is larger than size.
            pad_size = np.maximum(results['img_shape'][:2], self.size)
            padded_img = mmcv.impad(
                results['img'], shape=pad_size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str



class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


class AddArtifactualLines(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, prob, num_lines):
        self.prob = prob
        if type(num_lines)==int:
            assert len(num_lines)==2
            self.num_lines = (num_lines[0], num_lines[1])
        else:
            assert type(num_lines)==tuple
            self.num_lines = num_lines
        

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        img = results['img']
        assert len(img.shape)==3
        if random.random()<self.prob:
            num_lines = random.randint(self.num_lines[0], self.num_lines[1])
            lines = random.randint(1, img.shape[0], size=num_lines)
            for l in lines:
                img[l-1:l+1] = np.random.uniform(-1, high=0, size=2*img.shape[1]*3).reshape(2,img.shape[1], 3)
            results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, num_lines_min={self.num_lines[0]}, num_lines_max={self.num_lines[1]}' 
        return repr_str
    
    
    

class GrayTo3Channels(object):
    """Repeats the channel three times.
    """

    def __init__(self):
        self.to_rgb = True


    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_channels_cfg' key is added into
                result dict.
        """
        #äprint(results.keys(), type(results['img']))
        #print(results['img'])
        results['img'] = mmcv.imconvert(results['img'], 'gray','rgb')
        results['img_channels_cfg'] = dict(to_rgb=self.to_rgb)
        results['img_shape'] = results['img'].shape
        
        img_norm_cfg = results.get('img_norm_cfg', None)
        m = img_norm_cfg.get('mean', (0,))
        s = img_norm_cfg.get('std', (1,))
        if len(m)==1:
            m = (m[0], m[0], m[0])
            s = (s[0], s[0], s[0])
        results['img_norm_cfg'] = dict(to_rgb=self.to_rgb, mean=m, std=s)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_rgb={self.to_rgb})'
        return repr_str




class Rerange(object):
    """Rerange the image pixel value.

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, results):
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img
        results['img_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str



class CLAHE(object):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def __call__(self, results):
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, '\
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str



class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        #print(results['img_info'],results['img_prefix'],results['scale_factor'])
        #print('0: ',img.shape)
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        #print('1: ',img_shape)
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

from collections.abc import Sequence
from mmcv.parallel import DataContainer as DC
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')
    
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        if 'valid_pseudo_mask' in results:
            results['valid_pseudo_mask'] = DC(
                to_tensor(results['valid_pseudo_mask'][None, ...]), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__

class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


class RandomRotate(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str



class RGB2Gray(object):
    """Convert RGB image to grayscale image.

    This transform calculate the weighted mean of input image channels with
    ``weights`` and then expand the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transforming. Default: None.
        weights (tuple[float]): The weights to calculate the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights

    def __call__(self, results):
        """Call function to convert RGB image to grayscale image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with grayscale image.
        """
        img = results['img']
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'
        return repr_str



class AdjustGamma(object):
    """Using gamma correction to process the image.

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'



class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'



class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img'].astype(np.float32)
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str



class SpeckleNoise(object):
    def __init__(self, noise_rate, prob=.5):
        self.noise_rate = noise_rate # also known as scale
        self.rng = np.random.default_rng(1234)
        self.prob = prob

    def __call__(self, results):
        """
        :param img: numpy array of shape (H,W,C)
        :return:
        """
        # get noise from negative exponential distribution
        # scale = 1 / lambda
        if self.rng.random() < self.prob:
            img = results['img']
            speckle_noise = self.rng.exponential(scale=self.noise_rate, size=img.shape)
            img = img * speckle_noise
            results['img'] = img
            return results
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(noise_rate={0})'.format(self.noise_rate)



class RescaleMinMax(object):
    """Non linear intensity shift
    
    Used in the DRUNET paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6033560/
    The paper does not contain concrete details about the benefits of this transformation.
    """
    def __init__(self, min_val=0,max_val=1):
        self.min_val = min_val
        self.max_val = max_val
        #assert min_val==0,f"min_val must be {min_val}"

    def __call__(self, results):
        #print(results['img'].min(), results['img'].max())
        
        img = results['img']
        # normalize to [0,1]
        img, _, _ = normalize(img)# - img.min()) / (img.max() - img.min())
        if self.max_val!=1 or self.min_val!=0:
            img = inv_normalize(img, self.min_val, self.max_val)
        results['img'] = img
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '(min={0},max={1})'.format(self.min_val, self.max_val)
    


class ClipPercentile(object):
    """ClipPercentile
    """
    def __init__(self, lower_perc=1,upper_perc=99):
        self.lower_perc = lower_perc 
        self.upper_perc = upper_perc
        #assert min_val==0,f"min_val must be {min_val}"

    def __call__(self, results):
        
        img = results['img']
        percs = np.percentile(img, [self.lower_perc, self.upper_perc])
        results['img'] = np.clip(img, percs[0], percs[1])
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '(min={0},max={1})'.format(self.lower_perc, self.upper_perc)
    


class NonLinearIntensityShift(object):
    """Non linear intensity shift
    
    Used in the DRUNET paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6033560/
    The paper does not contain concrete details about the benefits of this transformation.
    """
    def __init__(self, prob=.5):
        self.rng = np.random.default_rng(1234)
        self.prob = prob  

    def __call__(self, results):
        #print(results['img'].min(), results['img'].max())
        
        img = results['img']
        # normalize to [0,1]
        img, min_val, max_val = normalize(img)#(img - img.min()) / (img.max() - img.min())
        if self.rng.random() < self.prob:
            bias = self.rng.uniform(0, .1)
            scale = self.rng.uniform(0, .1)
            exponent = self.rng.uniform(0.6, 1.4)
            img = - bias + (bias + scale +1)*img**exponent
            img, _, _ = normalize(img)# this make the bias shift useless
        results['img'] = img
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '(prob={0})'.format(self.prob)
    

def normalize(img):
    max_val = np.max(img)
    min_val = np.min(img)
    img = (img -  min_val) / (max_val - min_val) 
    return img, min_val, max_val 

def inv_normalize(img, min_val, max_val):
    return img*(max_val-min_val)+min_val


class BrightnessGradient(object):
    """
    Home-OCT and Spectralis OCT have a different brightness characteristic.
    The brightness is mostly related to structures such as choroid, retina, glass body
    
    In Spectralis OCT the choroid and glass body are darker than in Home-OCT.
    
    The following function allows to change the brightness based on the y-pixel position.
    It was developed for changing the brightness of the spectralis data.
    """
    def __init__(self, prob=.5, y_poly=lambda y: 1 + np.random.uniform(1, 2)*y):
        self.rng = np.random.default_rng(1234)
        self.prob = prob  
        self.y_poly = y_poly

    def __call__(self, results):

        img = results['img']
        # normalize to [0,1]
        img, min_val, max_val = normalize(img)
        if self.rng.random() < self.prob:
            y = np.linspace(0,1,num=len(img))
            y = self.y_poly(y)[:,np.newaxis,np.newaxis]
            avg_stdAscan = np.mean(np.std(img, axis=0))
            dc = np.clip(np.squeeze(self.rng.normal(1)*avg_stdAscan + np.mean(img)),0,1)#np.random.uniform(0.1,.2)
            img, _, _ = normalize((img + dc)*y)
        results['img'] = inv_normalize(img, min_val, max_val)
        return results
    
    def __repr__(self):
        return self.__class__.__name__ + '(prob={0})'.format(self.prob)
    


class PatchWiseNoiseSwapping(object):
    """Swaps patch"""
    def __init__(self, noise_rate, patch_size ,prob=.5):
        self.noise_rate = noise_rate # also known as scale
        self.rng = np.random.default_rng(1234)
        self.prob = prob

    def __call__(self, results):
        """
        :param results: dict with keys 'img'
        :return:
        """
        # get noise from negative exponential distribution
        # scale = 1 / lambda
        if self.rng.random() < self.prob:
            img = results['img']
            speckle_noise = self.rng.exponential(scale=self.noise_rate, size=img.shape)
            img = img * speckle_noise
            results['img'] = img
            return results
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(noise_rate={0})'.format(self.noise_rate)
