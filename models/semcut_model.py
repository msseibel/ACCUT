import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class SemCUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='1,7,11,16', help='compute NCE loss on which layers')
        parser.add_argument('--seg_layers', type=str, default='0,1,2', help='compute segmentation loss on which layers')
        parser.add_argument('--n_levels', type=int, default=3, help='number of resolutions in the encoder and decoders')
        parser.add_argument('--n_classes', type=int, default=5, help='number of segmentation classes')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--lambda_seg_src', type=float, default=0.0, help='Use segmentation loss in source domain.')
        parser.add_argument('--lambda_seg_con', type=float, default=0.0, help='Enforces weaker segmentation head invariance. Use segmentation consistency loss between src and fake tgt ')
        parser.add_argument('--lambda_seg_tgt', type=float, default=0.0, help='Use segmentation loss from target domain.')
        parser.add_argument('--lambda_seg_con_tgt', type=float, default=0.0, help='Use segmentation loss from target domain.')
        parser.add_argument('--lambda_mres_seg_con', type=float, default=0.0, help='Enforces strong segmentation head invariance. Use multiple resolution segmentation consistency loss between src and fake tgt (or tgt and tgtidt)')
        #parser.add_argument('--connect_fun', type=str, default='cat', help='how to connect the multi resolution features to the decoder(s)')
        parser.add_argument('--ignore_index', type=int, default=5, help='ignore index for segmentation loss')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--D_iters', type=int, default=1, help='number of discriminator updates per generator update.')
        parser.add_argument('--input_noise', type=util.str2bool, nargs='?', const=True, default=False, help='add noise to the input of the discriminator')
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.use_seg_decoder = self.opt.lambda_seg_src > 0.0 or self.opt.lambda_seg_tgt > 0.0 or self.opt.lambda_seg_con > 0.0 or self.opt.lambda_seg_con_tgt > 0.0 or self.opt.lambda_mres_seg_con > 0.0
        if not self.use_seg_decoder:
            opt.connect_fun = None

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        if self.use_seg_decoder:
            self.loss_names += ['seg']
        if self.isTrain and opt.gan_mode =='wgangp':
            self.loss_names += ['grad_pen']
        
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.seg_layers = [int(i) for i in self.opt.seg_layers.split(',')]
        self.D_updates_per_G = opt.D_iters

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.use_seg_decoder and self.isTrain:
            self.visual_names+=['mask_A']
            self.visual_names+=['pred_real_mask_A'] # segmentation based on real_A
            self.visual_names+=['mask_B']
            self.visual_names+=['pred_real_mask_B'] # segmentation based on real_B

        self.model_names = ['enc', 'decS']
        if self.use_seg_decoder:
            self.model_names += ['decM']
        if self.isTrain:
            self.model_names.extend(['F', 'D'])
            
        
        # input_nc, output_nc, ngf, net_enc, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], n_levels=2, opt=None
        self.netenc = networks.define_encoder(opt.input_nc, 'qwe', opt.ngf, opt.net_enc, opt.norm_enc, not opt.no_dropout, opt.init_type, opt.init_gain, 
                                               opt.no_antialias, self.gpu_ids, opt.n_levels, opt)
        self.netdecS = networks.define_decoder(
            'qwe', opt.output_nc, opt.ngf, opt.net_decS, opt.norm_dec, not opt.no_dropout, opt.init_type, opt.init_gain, 
                                                     opt.no_antialias_up, self.gpu_ids, opt.n_levels, opt.connect_fun, 'style', opt)
        if self.use_seg_decoder:
            self.netdecM = networks.define_decoder(
                'qwe', opt.n_classes, opt.ngf, opt.net_decM, opt.norm_dec, not opt.no_dropout, opt.init_type, opt.init_gain, 
                                                        opt.no_antialias_up, self.gpu_ids, opt.n_levels, opt.connect_fun, 'mask', opt)
        
        # define networks (both generator and discriminator)
        #self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, opt.input_noise, self.gpu_ids, opt)

            # define loss functions
            self.criterionSeg = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=opt.ignore_index).to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode, prob_label_noise=opt.prob_label_noise).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_enc = torch.optim.Adam(self.netenc.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_decS = torch.optim.Adam(self.netdecS.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            if self.use_seg_decoder:
                self.optimizer_decM = torch.optim.Adam(self.netdecM.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_enc)
            self.optimizers.append(self.optimizer_decS)
            if self.use_seg_decoder:
                self.optimizers.append(self.optimizer_decM)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        # compute fake images: G(A)
        self.forward()  # prepares da     
        self.forward_style()
        
        if self.opt.isTrain:
            if self.opt.gan_mode!='wgangp':
                self.compute_D_loss().backward()                  # calculate gradients for D
            
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
    
    def compute_seg_loss(self, real_mask_A, real_mask_B=None, fake_B_mask=None, idt_B_mask=None):
        seg_loss = 0.
        
        # supervised segmentation loss with source domain labels
        if self.opt.lambda_seg_src > 0.0:
            sup_seg_loss_src =  self.criterionSeg(real_mask_A, self.mask_A[:,0])
            seg_loss += sup_seg_loss_src

        # supervised segmentation loss with target domain labels
        if self.opt.lambda_seg_tgt > 0.0:
            sup_seg_loss_tgt = self.criterionSeg(real_mask_B, self.mask_B[:,0])
            seg_loss += sup_seg_loss_tgt
            
        # source segmentation consistency loss
        if self.opt.lambda_seg_con > 0.0:
            seg_con_loss_src = torch.mean((real_mask_A - fake_B_mask)**2)
            seg_loss += (self.opt.lambda_seg_con * seg_con_loss_src)
            
        if self.opt.lambda_mres_seg_con > 0.0:
            raise NotImplementedError
        
        # target segmentation consistency loss
        if self.opt.lambda_seg_con_tgt > 0.0:
            seg_con_loss_tgt = torch.mean((real_mask_B - idt_B_mask)**2)
            seg_loss += (self.opt.lambda_seg_con_tgt * seg_con_loss_tgt)
        return seg_loss
        
    def optimize_seg(self, real_mask_A, real_mask_B=None, fake_B_mask=None, idt_B_mask=None):
        if self.use_seg_decoder:
            self.optimizer_decM.zero_grad()
            self.optimizer_enc.zero_grad()
            self.loss_seg = self.compute_seg_loss(real_mask_A, real_mask_B, fake_B_mask, idt_B_mask)
            self.loss_seg.backward(retain_graph=False) # the encoder graph is still needed. (unfortunately, this retains also the graph of the segmentation decoder)
            if self.use_seg_decoder: 
                self.optimizer_decM.step()
            self.optimizer_enc.step()
    
    def optimize_style(self):
        ## update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward() 
        self.optimizer_D.step()
#
        ## update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_enc.zero_grad()
        self.optimizer_decS.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_enc.step()
        self.optimizer_decS.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            
    
    def optimize_parameters(self):
        self.forward()
        # update semantic decoder and retain encoder graph
        real_latent, mres_enc = self.encode_real()
        
        if self.use_seg_decoder:
            real_mask, _ = self.decode_seg(real_latent, mres_enc)
            pred_real_mask = torch.argmax(real_mask, dim=1, keepdim=True) # for display
            self.pred_real_mask_A = pred_real_mask[:self.real_A.size(0)]
            self.pred_real_mask_B = pred_real_mask[self.real_A.size(0):]
            real_mask_A = real_mask[:self.real_A.size(0)]
            real_mask_B = real_mask[self.real_A.size(0):]

        
        self.optimize_seg(real_mask_A, real_mask_B)
        self.forward_style()
        self.optimize_style()
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.mask_A = input['mask_A' if AtoB else 'mask_B'].to(self.device)
        self.mask_B = input['mask_B' if AtoB else 'mask_A'].to(self.device)
        self.image_paths = input['name']#input['A_paths' if AtoB else 'B_paths']
        self.style_path = input.get('style_name','')

    def encode_real(self):
        real_latent, mres_enc = self.netenc(
            self.real, 
            layers=self.nce_layers) 
        # set output attributes
        self.real_mres_enc_A = [el[:self.real_A.size(0)] for el in mres_enc]
        if self.opt.nce_idt:
            self.real_mres_enc_B = [el[self.real_A.size(0):] for el in mres_enc]
        return real_latent, mres_enc
    
    def encode_fake(self):
        # Encoding for contrastive loss and identity loss
        tgt_latents, mres_enc_fake = self.netenc(self.fake, layers=self.nce_layers)
        self.fake_mres_enc_B = [el[:self.real_A.size(0)] for el in mres_enc_fake]
        if self.opt.nce_idt:
            self.idt_mres_enc_B =  [el[self.real_A.size(0):] for el in mres_enc_fake]
        return tgt_latents, mres_enc_fake
    
    def decode_seg(self, real_latent, mres_enc):
        real_mask, mres_mask = self.netdecM(real_latent, mres_enc[:-1][::-1], layers=self.seg_layers) 
        return real_mask, mres_mask
    
    def decode_style(self, real_latent, mres_mask=None):
        # stop gradients to segmentation network
        if mres_mask is not None:
            mres_mask_detached = [el.clone().detach() for el in mres_mask]
        else:
            mres_mask_detached = [None for _ in range(len(self.seg_layers))]
        fake = self.netdecS(real_latent, mres_mask_detached)  
        return fake
    
    def forward_style(self):
        real_latent, mres_enc = self.encode_real()
        if self.use_seg_decoder:
            with torch.no_grad():
                _, mres_mask = self.decode_seg(real_latent, mres_enc)
                #mres_mask = [torch.zeros_like(el) for el in mres_mask]
        else:
            mres_mask = None

        self.fake = self.decode_style(real_latent, mres_mask)
        
        self.fake_B = self.fake[:self.real_A.size(0)] # required for the GAN loss
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        
        if self.opt.isTrain:
            self.encode_fake()
        return self.fake
        
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if hasattr(self, 'fake_B'):del self.fake_B
        if hasattr(self, 'idt_B'):del self.idt_B
        
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.mask = torch.cat((self.mask_A, self.mask_B), dim=0) if (self.opt.lambda_seg_tgt > 0.0) and self.opt.isTrain else self.mask_A
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
        
        # inference mode
        if not self.opt.isTrain:
            real_latent, mres_enc = self.encode_real()
            if self.use_seg_decoder:
                real_mask, _ = self.decode_seg(real_latent, mres_enc)
                pred_real_mask = torch.argmax(real_mask, dim=1, keepdim=True) # for display
                self.pred_real_mask_A = pred_real_mask[:self.real_A.size(0)]
                self.pred_real_mask_B = pred_real_mask[self.real_A.size(0):]
                real_mask_A = real_mask[:self.real_A.size(0)]
                real_mask_B = real_mask[self.real_A.size(0):]
            self.forward_style()
        
        # self.real_mask, self.fake_mask
        #if self.opt.lambda_seg_con > 0.0: # segmentation consistency
        #    if self.opt.lambda_mres_seg_con > 0.0:
        #        self.fake_mask , self.fake_mres_mask = self.netdecM(self.tgt_latents, mres_enc_fake[:-1][::-1], layers=self.seg_layers)
        #    else:
        #        self.fake_mask  = self.netdecM(self.tgt_latents, mres_enc_fake[:-1][::-1], layers=False)
        #    self.fake_B_mask = self.fake_mask[:self.real_A.size(0)] # mask generated from src2tgt 
            


    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        if self.opt.gan_mode =='wgangp':
            self.loss_grad_pen, gradients = networks.cal_gradient_penalty(
                self.netD, self.real_B, self.fake_B, self.device, lambda_gp=10, type='mixed',constraint='equal')
            self.loss_grad_pen = self.loss_grad_pen
            self.loss_grad_pen.backward(retain_graph=True) 
             #self.loss_D = (self.loss_D_real + self.loss_D_fake ) * 0.5 # not sure about the sign of the gradient penalty. Perhaps it should be -gradient penalty
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        #elif self.opt.gan_mode in ['lsgan', 'vanilla', 'nonsaturating']:
        #    # combine loss and calculate gradients
        #    self.loss_D = (self.loss_D_fake + self.loss_D_real) *0.5
        #else:
        #    raise NotImplementedError
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            # real_latent_A, fake_latent_B (encoding based on the real_A and fake_B)
            self.loss_NCE = self.calculate_NCE_loss(self.real_mres_enc_A, self.fake_mres_enc_B) # seg_mask
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            # real_latent_B, idt_latent_B (encoding based on the real_B and idt_B)
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_mres_enc_B, self.idt_mres_enc_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, feat_k, feat_q):
        """
        feat_q: features of the source image
        feat_k: features of the translated image
        """
        n_layers = len(self.nce_layers)
        
        # Used for Fast-CUT        
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        # Extract patches
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids) # get 'patches' of the translated image from the same locations as feat_k_pool
        

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            # f_q: (B*num_patches, C) ???
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
