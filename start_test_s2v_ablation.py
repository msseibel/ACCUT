import subprocess
src = 'Spectralis'
tgt = 'Visotec'
version =14.1

#output_dir = 'bvm_baseline'
#output_dir = 'bvm_doubleseg'
#output_dir = 'bvm_targetseg'
#output_dir = 'bvm_sourceseg'

mode = 'target'
output_dir = f'bvm_ablationpermute_{mode}seg'

if output_dir == 'bvm_baseline':
    lambda_seg_src = '0.0'
else:
    lambda_seg_src = '1.0'
epoch = 250
#../../tmp/Spectralis_Visotec_v14.1/tiff/Spectralis
subprocess.run(['python', 'test.py',
"--content_path", f"../../tmp/Spectralis_Visotec_v{version}/tiff/{src}",
"--style_path", f"../../tmp/Spectralis_Visotec_v{version}/tiff/{tgt}" ,
#"--content",f"{src}",
#"--style", f"{tgt}",
"--name", f'spec2home_CUT_bvm_{mode}seg_lsgan',
"--model", 'semcutablation',
"--CUT_mode", "CUT",
"--dataset_mode", 'visotec_spectralis_ablation',
"--load_size", '512',
"--crop_size", '240',
"--gpu_ids", '0',
'--num_test', '10000000',
"--output_dir", f'{output_dir}',
"--input_nc", "1",
"--output_nc", "1",
"--load_src_seg", "True",
"--load_tgt_seg", "True",
"--epoch", f"{epoch}",
"--lambda_seg_src", lambda_seg_src,
])