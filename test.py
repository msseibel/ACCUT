"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from PIL import Image
import numpy as np
from pathlib import Path
import csv
import torch

tmp_dir = Path("../../tmp")


def append_row_to_csv(row, csv_file_path):    
    with open(csv_file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(row)

def get_eye(img_name):
    if 'OD' in img_name:
        return 'OD'
    elif 'OS' in img_name:
        return 'OS'
    else:
        return 'XX'
    
def info_from_img_name(img_name):
    # format of name: domain_subid_eye-001.png
    domain = img_name.split('_')[0]
    sub_id = img_name.split('_')[1]
    eye    = get_eye(img_name)
    slice_idx = img_name.split('.')[0].split('-')[-1]
    assert len(slice_idx) == 3, f"slice_idx: {slice_idx}, img_name: {img_name}" # We assume that the slice index is 3 digits long (padded with zeros)
    assert domain in ['RETOUCH-Cirrus2Topcon', 
                      'RETOUCH-Cirrus2Spectralis',
                      'RETOUCH-Topcon2Cirrus', 
                      'RETOUCH-Topcon2Spectralis', 
                      'RETOUCH-Spectralis2Cirrus', 
                      'RETOUCH-Spectralis2Topcon', 
                      'Visotec', 'Spectralis']
    return domain, sub_id, eye, slice_idx
    
def writeX_to_tiff(X_slice, domain, sub_id, eye, slice_idx, output_dir):
    os.makedirs(tmp_dir / output_dir / "tiff" / f'{domain}' / f'{sub_id}', exist_ok=True)
    #print('Save to', tmp_dir / output_dir / "tiff" / f'{domain}' / f'{sub_id}' / f"X-{eye}-{slice_idx}.tiff")
    if type(X_slice) == np.ndarray:
        X_slice = Image.fromarray(X_slice)
 
    if domain not in ['RETOUCH-Cirrus2Topcon', 
                      'RETOUCH-Cirrus2Spectralis',
                      'RETOUCH-Topcon2Cirrus', 
                      'RETOUCH-Topcon2Spectralis', 
                      'RETOUCH-Spectralis2Cirrus', 
                      'RETOUCH-Spectralis2Topcon', 
                      'Visotec', 'Spectralis']:
        raise ValueError(f'Unknown domain {domain}')
    
    X_slice.save(tmp_dir / output_dir / "tiff" / f'{domain}' / f'{sub_id}' / f"X-{eye}-{slice_idx}.tiff")


def writeY_to_numpy(Y_slice, domain, sub_id, eye, slice_idx, output_dir, src_tgt):
    os.makedirs(tmp_dir / output_dir / "npy" / f'{domain}' / f'{sub_id}', exist_ok=True)
    if type(Y_slice) == torch.Tensor:
        Y_slice = Y_slice.detach().cpu().numpy()
    elif type(Y_slice) == Image.Image:
        Y_slice = np.array(Y_slice)
    elif type(Y_slice) != np.ndarray:
        assert type(Y_slice) == np.ndarray, f"Y_slice is not a numpy array but {type(Y_slice)}"
        
    if domain not in ['RETOUCH-Cirrus2Topcon', 
                      'RETOUCH-Cirrus2Spectralis',
                      'RETOUCH-Topcon2Cirrus', 
                      'RETOUCH-Topcon2Spectralis', 
                      'RETOUCH-Spectralis2Cirrus', 
                      'RETOUCH-Spectralis2Topcon', 
                      'Visotec', 'Spectralis']:
        raise ValueError(f'Unknown domain {domain}')
    np.save(tmp_dir / output_dir / "npy" / f'{domain}' / f'{sub_id}' / f"ACCUTY-{eye}-{slice_idx}-{src_tgt}.npy", Y_slice)

def normalize_volumes(output_dir, domain):
    dataset_dir = tmp_dir / output_dir / "tiff" / f'{domain}'
    if SUBJECTS is None:
        SUBJECTS = os.listdir(dataset_dir)
    for sub_id in SUBJECTS:#os.listdir(dataset_dir):
        for eye in ['OD', 'OS', 'XX']:
            # list all files which contain the eye and the sub_id
            files = [f for f in os.listdir(dataset_dir / sub_id) if eye in f]
            if len(files) == 0:
                continue
            # load all slices of the subject and the eye
            slices = []
            slice_idxs = []
            for f in files:
                slices.append(np.array(Image.open(dataset_dir / sub_id / f)))
                slice_idx = f.split('-')[-1].split('.')[0]
                slice_idxs.append(slice_idx)
                
            # normalize the slices
            slices = np.array(slices)
            slices = (slices - np.mean(slices)) / np.std(slices)
            # save the normalized slices
            for idx in range(len(slices)):
                writeX_to_tiff(slices[idx], domain, sub_id, eye, slice_idxs[idx], output_dir)

def resize(X, ori_size_wh, method=Image.BICUBIC):
    X_slice = Image.fromarray(X)
    # resize back to original size
    if np.any(X_slice.size!=ori_size_wh):
        X_slice = X_slice.resize(ori_size_wh, method)
    return X_slice

if __name__ == '__main__':
    SUBJECTS = ['081']
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    output_dir = opt.output_dir
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    
    web_dir = Path("../../tmp") / os.path.join(output_dir, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    csv_file_path = os.path.join(web_dir, f'{opt.name}_mapping.csv')
    print('Csv file ', csv_file_path)
    with open(csv_file_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['src_img_path', 'style_img_path', 'ablation_img_path'])

    for i, data in enumerate(dataset):
        print(data['A_paths'])
        print(data['name'])
        if not '_81_' in data['name'][0]:
            continue
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        row = []
        img_path   = model.get_image_paths()     # get image paths
        style_path = model.get_style_paths()     # get image paths
        row+=[img_path[0]]
        row+=[style_path[0]]
        if hasattr(model, 'get_ablation_paths'):
            ablation_img_path = model.get_ablation_paths()
            row+=[ablation_img_path[0]]
        #print(row)
        append_row_to_csv(row, csv_file_path=csv_file_path)

        #if i % 5 == 0:  # save images to an HTML file
        #    print('processing (%04d)-th image... %s' % (i, img_path))
        #save_images(webpage, visuals, img_path, width=opt.display_winsize)
        domain, sub_id, eye, slice_idx = info_from_img_name(img_path[0])
        X = model.fake_B[0,0].detach().cpu().numpy()
        
        ori_size_wh = tuple(np.concatenate(data['ori_size']))
        ori_size_wh_tgt = tuple(np.concatenate(data['ori_size_tgt']))
        print('Original size:', ori_size_wh, 'Target size:', ori_size_wh_tgt)
        #break
        X_slice = resize(X, ori_size_wh)
        
        writeX_to_tiff(X_slice, domain, sub_id, eye, slice_idx, output_dir)
        if opt.save_seg:
            Ysrc = model.pred_real_mask_A
            print('Ysrc shape before resize', Ysrc.shape)
            Ysrc = Ysrc[0,0].detach().cpu().numpy().astype(np.uint8)
            Ysrc = resize(Ysrc, ori_size_wh, method=Image.NEAREST)
            #print('Ysrc shape after resize', Ysrc.shape)
            writeY_to_numpy(Ysrc, domain, sub_id, eye, slice_idx, output_dir, 'src')
            
            #break
            #Ytgt = model.pred_real_mask_B
            #print('Ytgt shape before resize', Ytgt.shape)
            #Ytgt = Ytgt[0,0].detach().cpu().numpy().astype(np.uint8)
            #Ytgt = resize(Ytgt, ori_size_wh_tgt, method=Image.NEAREST)
            #print('Ytgt shape after resize', Ytgt.shape)
            #writeY_to_numpy(Ytgt, domain, sub_id, eye, slice_idx, output_dir, 'tgt')
            #print('Saved segmentation to ', tmp_dir / output_dir / "npy" / f'{domain}' / f'{sub_id}' / f"ACCUTY-{eye}-{slice_idx}-src.npy")
    normalize_volumes(output_dir, domain)
    #webpage.save()  # save the HTML
