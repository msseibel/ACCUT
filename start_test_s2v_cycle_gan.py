import subprocess
src = 'Spectralis'
tgt = 'Visotec'
version =14.1

#output_dir = 'bvm_baseline'
#output_dir = 'bvm_doubleseg'
#output_dir = 'bvm_targetseg'
output_dir = 'cycle_gan'
epoch = 350
#../../tmp/Spectralis_Visotec_v14.1/tiff/Spectralis
subprocess.run(['python', 'test.py',
"--content_path", f"../../tmp/Spectralis_Visotec_v{version}/tiff/{src}",
"--style_path", f"../../tmp/Spectralis_Visotec_v{version}/tiff/{tgt}" ,
#"--content",f"{src}",
#"--style", f"{tgt}",
"--name", f'spec2home_cg',
"--model", 'cycle_gan',
"--dataset_mode", 'visotec_spectralis_cg',
"--load_size", '512',
"--crop_size", '240',
"--gpu_ids", '0',
'--num_test', '10000000',
"--output_dir", f'{output_dir}-{epoch}',
"--input_nc", "1",
"--output_nc", "1",
"--epoch", f"{epoch}",
"--save_seg", "False"
])