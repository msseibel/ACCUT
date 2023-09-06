import os
from data.base_dataset import BaseDataset, get_transform
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


# get recursively all tiff files which start with X under a directory
def get_tiff_files(directory, startswith, min_samples=None):
    print('Start loading files...', directory)
    files = {'Spectralis':[], 'Visotec':[]}
    device_counter = {'Spectralis':0, 'Visotec':0}
    for root, _, fnames in tqdm.tqdm(sorted(os.walk(directory))):
        for fname in fnames:
            if fname.startswith(startswith) and fname.endswith('.tiff'):
                path = os.path.join(root, fname)
                for device in device_counter.keys():
                    if device in path.split('/'):
                        device_counter[device] += 1

                        files[device].append(path)
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


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class VisotecSpectralisDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.content_path
        self.dir_B = opt.style_path
        if opt.mixed:
            files = get_tiff_files(self.dir_A,'X', 'balanced')
            files = files['Spectralis'].tolist() + files['Visotec'].tolist()
            self.A_paths = files.copy()#sorted(make_dataset(self.dir_A, opt.max_dataset_size))
            self.B_paths = files.copy()#sorted(get_tiff_files(self.dir_B,'X'))#sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        else:
            filesA = get_tiff_files(self.dir_A,'X', None)
            filesB = get_tiff_files(self.dir_B,'X', None)
            self.A_paths = np.concatenate(list(filesA.values()))#.copy()#sorted(make_dataset(self.dir_A, opt.max_dataset_size))
            self.B_paths = np.concatenate(list(filesB.values()))#.copy()#sorted(get_tiff_files(self.dir_B,'X'))#sorted(make_dataset(self.dir_B, opt.max_dataset_size))
            
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(self.opt)
        print('crop_pos' in self.opt)

        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

    def __getitem__(self, index):
        if self.opt.isTrain:
            index_A = index
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_A = index
            index_B = index % self.B_size # iterate through B 


        A_path = self.A_paths[index_A]
        A_img = Image.open(A_path)
        A_img = Image.fromarray(normalize(np.array(A_img))*255)
        A_img = A_img.convert('RGB')
        A = self.transform_A(A_img)


        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path)
        B_img = Image.fromarray(normalize(np.array(B_img))*255)
        B_img = B_img.convert('RGB')
        B = self.transform_B(B_img)




        name_A = '_'.join(A_path.split('/')[-3:])#os.path.basename(A_path)
        name_B = '_'.join(B_path.split('/')[-3:])#os.path.basename(B_path)
        if self.opt.isTrain:
            name = name_B[:name_B.rfind('.')] + '_' + name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
        else:
            name = name_A[:name_A.rfind('.')] + name_A[name_A.rfind('.'):]
        result = {'A': A, 'B': B, 'name': name, 'style_name':name_B, 'ori_size':A_img.size, 'A_paths': A_path, 'B_paths': B_path}
        return result

    def __len__(self):
        if self.opt.isTrain:
            return self.A_size
        else:
            return self.A_size#min(self.A_size * self.B_size, self.opt.num_test)