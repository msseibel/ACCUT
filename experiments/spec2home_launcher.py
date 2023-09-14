from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def common_options(self):
        version = 14.1
        return [
            # Command 0
            Options(
                #dataroot
                content_path=f"../HRDA/data/Spectralis_Visotec_v{version}/Spectralis/img_dir/train/",
                style_path=f"../HRDA/data/Spectralis_Visotec_v{version}/Visotec/img_dir/train/",
                name="spec2home_CUT",
                dataset_mode="visotec_spectralis",
                CUT_mode="CUT",
                input_nc=1,
                output_nc=1,
            ),
            # Command 1
            Options(
                content_path=f"../HRDA/data/Spectralis_Visotec_v{version}/Spectralis/img_dir/train/",
                style_path=f"../HRDA/data/Spectralis_Visotec_v{version}/Visotec/img_dir/train/",
                name="spec2home_FastCUT",
                CUT_mode="FastCUT",
            ),
            Options(
                #dataroot
                content_path=f"../HRDA/data/Spectralis_Visotec_v{version}/Spectralis/img_dir/train/",
                style_path=f"../HRDA/data/Spectralis_Visotec_v{version}/Visotec/img_dir/train/",
                name="spec2home_UnbiasedCUT",
                dataset_mode="visotec_spectralis",
                CUT_mode="UnbiasedCUT",
                input_nc=1,
                output_nc=1,
            ),
        ]

    def commands(self):
        return ["python train.py " + str(opt) for opt in self.common_options()]

    def test_commands(self):
        # RussianBlue -> Grumpy Cats dataset does not have test split.
        # Therefore, let's set the test split to be the "train" set.
        return ["python test.py " + str(opt.set(phase='train')) for opt in self.common_options()]
