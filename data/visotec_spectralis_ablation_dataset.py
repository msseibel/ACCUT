import os
from data.base_dataset import BaseDataset, get_transform, get_transforms_dict, get_transforms_dict_test
from data.image_folder import make_dataset
from PIL import Image
import random
import torch.backends.cudnn as cudnn
from PIL import ImageFile
import numpy as np
import tqdm

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def parse_directory(directory, startswith):
    files = {'Spectralis':[], 'Visotec':[]}
    device_counter = {'Spectralis':0, 'Visotec':0}
    
    if directory.split('/')[-1] in ['Spectralis', 'Visotec']:
        device = directory.split('/')[-1]
        for sub in os.listdir(directory):
            sub_dir = os.path.join(directory, sub)
            if not os.path.isdir(sub_dir):
                continue
            for img in os.listdir(sub_dir):
                if img.startswith(startswith):
                    path = os.path.join(directory, sub, img)
                    device_counter[device] += 1
                    files[device].append(path)

    else: #slower method
        for root, _, fnames in tqdm.tqdm(sorted(os.walk(directory))):
            for fname in fnames:
                if fname.startswith(startswith) and fname.endswith('.tiff'): # 
                    path = os.path.join(root, fname)
                    for device in device_counter.keys():
                        if device in path.split('/'):
                            device_counter[device] += 1
                            files[device].append(path)
    return files, device_counter

# get recursively all tiff files which start with X under a directory
def get_tiff_files(directory, startswith, min_samples=None):
    print('Start loading files...', directory)
    #files = {'Spectralis':[], 'Visotec':[]}
    #device_counter = {'Spectralis':0, 'Visotec':0}
    files, device_counter = parse_directory(directory, startswith)

    if min_samples=='balanced':
        rng = np.random.RandomState(0)
        common_min = min(device_counter.values())
        print('common_min', common_min)
        files['Spectralis'] = rng.choice(files['Spectralis'], size=common_min, replace=False)
        files['Visotec'] = rng.choice(files['Visotec'], size=common_min, replace=False)
        assert len(files['Spectralis']) == len(files['Visotec'])
    elif type(min_samples)==int:
        assert len(files['Spectralis']) >= min_samples
        assert len(files['Visotec']) >= min_samples
    return files

import torch
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class VisotecSpectralisAblationDataset(BaseDataset):
    """
    Generator tanh maps values to -1, 1 range. Ensure that the dataset values are in the same range.
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.content_path
        self.dir_B = opt.style_path

        random.seed(9001)
        # Some images don't have an associated segmentation. 
        # startswith='Y' ensures that we only load images with a segmentation
        image_keys_src = ['img']
        if opt.load_src_seg:
            startswith_src = 'Y'
            image_keys_src.append('gt_semantic_seg')
        else:
            startswith_src = 'X'
        
        image_keys_tgt = ['img']
        if opt.load_tgt_seg:
            startswith_tgt = 'Y'
            image_keys_tgt.append('gt_semantic_seg')
        else:
            startswith_src = 'X'
        
        filesA = get_tiff_files(self.dir_A,startswith_src, None)
        filesB = get_tiff_files(self.dir_B,startswith_tgt, None)
        
        self.A_paths = np.concatenate(list(filesA.values()))#.copy()#sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = np.concatenate(list(filesB.values()))#.copy()#sorted(get_tiff_files(self.dir_B,'X'))#sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        # If image name startswith 'Y' then we have to map the image name back to 'X' due to further processing
        self.A_paths = [el.replace('/Y', '/X') for el in self.A_paths]
        self.B_paths = [el.replace('/Y', '/X') for el in self.B_paths]

        print('Num Files A: ', len(self.A_paths))
        print('Num Files B: ', len(self.B_paths))
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(self.opt)
        print('crop_pos' in self.opt)
        if self.opt.isTrain:
            self.transform_A = get_transforms_dict(
                dict(dataset=self.dir_A.split('/')[-1].lower(), 
                     image_keys=image_keys_src))#get_transform(self.opt)
            self.transform_B = get_transforms_dict(
                dict(dataset=self.dir_B.split('/')[-1].lower(), image_keys=image_keys_tgt))#get_transform(self.opt)
        else:
            self.transform_A = get_transforms_dict_test(
                dict(dataset=self.dir_A.split('/')[-1].lower(), image_keys=image_keys_src))
            self.transform_B = get_transforms_dict_test(
                dict(dataset=self.dir_B.split('/')[-1].lower(), image_keys=image_keys_tgt))
            
    def crop_pad(self, img, output_size):
        # Get the shape of the image
        img_shape = img.shape

        # Create a zero array with output_size
        output_img = np.zeros(output_size)

        # Calculate the coordinates to place the original image
        pad_coords = [(o - i) // 2 for o, i in zip(output_size, img_shape)]
        end_coords = [p + i for p, i in zip(pad_coords, img_shape)]

        # Crop if img_shape > output_size, else pad
        slices_input = [slice(max(0, -pad), min(i, o-pad)) for pad, i, o in zip(pad_coords, img_shape, output_size)]
        slices_output = [slice(max(0, pad), min(o, pad+i)) for pad, i, o in zip(pad_coords, img_shape, output_size)]

        # Place the image in the output_img array
        output_img[slices_output] = img[slices_input]

        return output_img
            
    def __getitem__(self, index):
        if self.opt.isTrain:
            index_A = index
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_A = index
            index_B = index % self.B_size # iterate through B 
            index_A_rand = random.randint(0, self.A_size - 1)


        A_path = self.A_paths[index_A]
        A_path_mask = A_path.replace('/X', '/Y')
        A_img  = Image.open(A_path)
        A_mask = Image.open(A_path_mask)

        B_path = self.B_paths[index_B]
        B_path_mask = B_path.replace('/X', '/Y')
        B_img = Image.open(B_path)
        B_mask = Image.open(B_path_mask)
        
        A_path_rand = self.A_paths[index_A_rand]
        A_path_mask_rand = A_path_rand.replace('/X', '/Y')
        A_img_rand  = Image.open(A_path_rand)
        A_mask_rand = Image.open(A_path_mask_rand)
        
        resultsA = {'img': np.array(A_img), 
                    'gt_semantic_seg': np.array(A_mask), 
                    'seg_fields': ['gt_semantic_seg'],
                    'ori_size':np.flip(A_img.size)}
        
        resultsA_rand = {'img': np.array(A_img_rand), 
                    'gt_semantic_seg': np.array(A_mask_rand), 
                    'seg_fields': ['gt_semantic_seg'],
                    'ori_size':np.flip(A_img_rand.size)}
        resultsA_rand['img'] = self.crop_pad(resultsA_rand['img'], output_size=resultsA['img'].shape)
        
        resultsB = {'img': np.array(B_img),
                    'gt_semantic_seg': np.array(B_mask), 
                    'seg_fields': ['gt_semantic_seg'],
                    'ori_size':np.flip(B_img.size)}
        
        resultsA = self.transform_A(resultsA) 
        resultsB = self.transform_B(resultsB) 
        results_rand = self.transform_A(resultsA_rand)
        
        A_rand = results_rand['img']
        mask_A_rand = results_rand['gt_semantic_seg']
        mask_A_rand[mask_A_rand==255] = 5 # remap index ignore to max(classes) + 1
        
        A = resultsA['img']
        mask_A = resultsA['gt_semantic_seg']
        mask_A[mask_A==255] = 5 # remap index ignore to max(classes) + 1
        
        B = resultsB['img']
        mask_B = resultsB['gt_semantic_seg']
        mask_B[mask_B==255] = 5 # remap index ignore to max(classes) + 1
        
            
        name_A = '_'.join(A_path.split('/')[-3:])#os.path.basename(A_path)
        name_B = '_'.join(B_path.split('/')[-3:])#os.path.basename(B_path)
        if self.opt.isTrain:
            name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
        else:
            name = name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
            #name_B = name_B.split('../../tmp/')[-1]
        A_path = A_path.split('../../tmp/')[-1]
        
        
        print(name)
        
        result = {'A': A, 
                  'A_rand': A_rand,
                  'B': B, 
                  'mask_A': mask_A.type(torch.long), 
                  'mask_B': mask_B.type(torch.long),
                  'name': name, 
                  'style_name':name_B, 
                  'ori_size':A_img.size, 
                  'ablation_img_path': A_path_rand,
                  'A_paths': A_path, 
                  'B_paths': B_path}
        return result

    def __len__(self):
        if self.opt.isTrain:
            return self.A_size
        else:
            return self.A_size#min(self.A_size * self.B_size, self.opt.num_test)