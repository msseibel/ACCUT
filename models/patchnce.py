from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0] # B * num_patches_per_img
        dim = feat_q.shape[1] # num_features
        feat_k = feat_k.detach()

        # pos logit, calc similarity between query and key (src and tgt patches)
        l_pos = torch.bmm(
            # (B * num_patches_per_img, 1, num_features) * (B * num_patches_per_img, num_features, 1) -> (B * num_patches_per_img, 1, 1)
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)) # (num_patches, 1, 1)
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size (unpack (B*num_patches_per_img, C) -> (B, num_patches_per_img, C))
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # matmul along feature axis -> (batch_size, npatches, npatches)

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :] # (1, npatches, npatches)
        l_neg_curbatch.masked_fill_(diagonal, -10.0) # (batch_size, npatches, npatches)
        l_neg = l_neg_curbatch.view(-1, npatches) # (num_patches, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T # (num_patches, 1+npatches) -> first column is pos, rest are negs

        # this vector indicates which entry of out is the positive sample (its always the first column)
        y_cat = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        loss = self.cross_entropy_loss(out, y_cat)
        return loss

def circulant(tensor, dim):
    """get a circulant version of the tensor along the {dim} dimension.
    
    The additional axis is appended as the last dimension.
    E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))

# This does not work
class UnbiasedPatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.max_classes = 5 # todo move to opt
    
    def weighted_nce(self, feat_q, feat_k, weights=None):
        """Implements a weighted nce loss, where weights is a adjacency matrix of size (batch_size, npatches, npatches)
        that determines decreases the logits for negative samples. This allows us to incorporate a prior knowledge about the
        matching of the patches.
        """
        num_patches = feat_q.shape[0] # B * num_patches_per_img
        dim = feat_q.shape[1] # num_features
        feat_k = feat_k.detach()

        # pos logit, calc similarity between query and key (src and tgt patches)
        l_pos = torch.bmm(
            # (B * num_patches_per_img, 1, num_features) * (B * num_patches_per_img, num_features, 1) -> (B * num_patches_per_img, 1, 1)
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)) # (num_patches, 1, 1)
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size (unpack (B*num_patches_per_img, C) -> (B, num_patches_per_img, C))
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # matmul along feature axis -> (batch_size, npatches, npatches)
        if weights is not None:
            # increasing the denominator in the NCE loss 
            # -> the network has 2 options to counteract this:
            # 1. increase numerator similarity
            # 2. decrease denominator similarity beyond our weighting intervention
            l_neg_curbatch += weights 

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :] # (1, npatches, npatches)
        l_neg_curbatch.masked_fill_(diagonal, -10.0) # (batch_size, npatches, npatches)
        l_neg = l_neg_curbatch.view(-1, npatches) # (num_patches, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T # (num_patches, 1+npatches) -> first column is pos, rest are negs

        # this vector indicates which entry of out is the positive sample (its always the first column)
        y_cat = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        loss = self.cross_entropy_loss(out, y_cat)
        return loss


    def vanilla_nce(self, feat_q, feat_k):
        num_patches = feat_q.shape[0] # B * num_patches_per_img
        dim = feat_q.shape[1] # num_features
        feat_k = feat_k.detach()

        # pos logit, calc similarity between query and key (src and tgt patches)
        l_pos = torch.bmm(
            # (B * num_patches_per_img, 1, num_features) * (B * num_patches_per_img, num_features, 1) -> (B * num_patches_per_img, 1, 1)
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)) # (num_patches, 1, 1)
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size (unpack (B*num_patches_per_img, C) -> (B, num_patches_per_img, C))
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # matmul along feature axis -> (batch_size, npatches, npatches)

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        # large negative values will not be considered as potential positive candidates.
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :] # (1, npatches, npatches)
        l_neg_curbatch.masked_fill_(diagonal, -10.0) # (batch_size, npatches, npatches)
        l_neg = l_neg_curbatch.view(-1, npatches) # (num_patches, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T # (num_patches, 1+npatches) -> first column is pos, rest are negs

        # this vector indicates which entry of out is the positive sample (its always the first column)
        y_cat = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        loss = self.cross_entropy_loss(out, y_cat)
        return loss

    def check_use_vanilla(self, classes):
        is_single_class = False
        for b_classes in classes:
            if len(torch.unique(b_classes)) == 1:
                is_single_class = True
                break
        return is_single_class
            
    def forward(self, features_src, features_tgt, classes=None, num_pairs_should=None, weights=None):
        if weights is not None:
            assert classes is None
            assert num_pairs_should is None
            return self.weighted_nce(features_src, features_tgt, weights=weights)

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size
        
        if classes is None:
            return self.vanilla_nce(features_src, features_tgt)
        
        
        assert features_src.isnan().sum() == 0, "features_src contains NaN values"
        assert features_tgt.isnan().sum() == 0, "features_tgt contains NaN values"
        #num_patches = features_src.shape[0]
        dim = features_src.shape[1] # num_features
    
        samples_ignore = (classes == 255).squeeze()
        # if more than 90% of samples are ignored, use vanilla nce
        num_samples_ignore = samples_ignore.sum()
        if  num_samples_ignore>= int(len(samples_ignore)*0.9):
            print(f'Out of {len(samples_ignore)} samples {num_samples_ignore} are ignored, using vanilla nce')
            return self.vanilla_nce(features_src, features_tgt)
        
        features_src = features_src[~samples_ignore]
        features_tgt = features_tgt[~samples_ignore]
        classes = classes[~samples_ignore]
        classes = classes.view(batch_dim_for_bmm, -1) # FIXME: samples_ignore makes it impossible to reshape the batch if batch_size > 1
        num_patches = features_src.shape[0]
        
        
        if self.check_use_vanilla(classes): # only one was sampled - need to use vanilla nce
            return self.vanilla_nce(features_src, features_tgt)
        

        logits_pos = torch.bmm(
                    # (B * num_patches_per_img, 1, num_features) * (B * num_patches_per_img, num_features, 1) -> (B * num_patches_per_img, 1, 1)
                    features_src.view(num_patches, 1, -1), features_tgt.view(num_patches, -1, 1)) # (num_patches, 1, 1)
        logits_pos = logits_pos.view(batch_dim_for_bmm, -1, 1)

        features_src = features_src.view(batch_dim_for_bmm, -1, dim)
        features_tgt = features_tgt.view(batch_dim_for_bmm, -1, dim)
        npatches = features_src.shape[1]
        
        # padding size
        if num_pairs_should is None:
            num_pairs_should = npatches

        out = []
        for b_logits_pos, b_feat_src, b_feat_tgt, b_classes in zip(logits_pos, features_src, features_tgt, classes):
            #b_logits_pos = 
            #print(b_logits_pos.shape, b_feat_src.shape, b_feat_tgt.shape, b_classes.shape)
            num_counts = torch.bincount(b_classes, minlength=self.max_classes)
            #print(num_counts)
            # critical problem: all samples in a batch may be from the same class
            num_pairs_is = -num_counts + 1 + npatches
            
            for c, counts_in_class in enumerate(num_counts):
                if counts_in_class == 0:
                    continue
                logits_pos_c = b_logits_pos[b_classes==c]
                l_neg = torch.mm(b_feat_src[b_classes==c], b_feat_tgt[b_classes!=c].T)
                smallest = torch.minimum(l_neg.min(), torch.tensor([-10.0]).to(l_neg.device))
                
                num_missing_neg_pairs = num_pairs_should - num_pairs_is[c]
                l_neg_pad = smallest * torch.ones(l_neg.shape[0], num_missing_neg_pairs).to(l_neg.device)
                
                out_c = torch.cat([logits_pos_c, l_neg, l_neg_pad],dim=1) / self.opt.nce_T
                out += [out_c]
        
        out = torch.cat(out, dim=0)
        y_cat = torch.zeros(out.size(0), dtype=torch.long, device=features_src.device)

        loss = self.cross_entropy_loss(out, y_cat)
        assert not torch.isnan(loss).any(), "Loss contains NaN values"
        assert loss.ge(0).all(), "Loss contains negative values"
        return loss
    
