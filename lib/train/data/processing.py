# In lib/train/data/processing.py

import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize

class BaseProcessing:
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.max_len_t = getattr(settings, 'num_template', 1)
        self.max_len_s = getattr(settings, 'num_search', 1)

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        # --- START OF REFACTORED LOGIC ---
        
        # 1. Keep a copy of the original (raw numpy) search images for the APN
        # We do this before any cropping or transformations.
        if 'search_images' in data:
            data['search_images_raw'] = [img.copy() for img in data['search_images']]

        # 2. Process template and search crops
        for s in ['template', 'search']:
            # Jitter the box
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            # This returns cropped numpy arrays
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
                data[s + '_images'], jittered_anno, data[s + '_anno'],
                self.search_area_factor[s], self.output_sz[s], masks=data[s + '_masks']
            )
            
            # Apply transforms (e.g., ToTensor, Normalize) to the CROPPED images
            # The output is a list of tensors
            data[s + '_images'], data[s + '_anno'], _, _ = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False
            )

        # 3. Pad all sequences to a fixed length and stack into single tensors
        for s in ['template', 'search']:
            max_len = self.max_len_t if s == 'template' else self.max_len_s
            
            for key in [f'{s}_images', f'{s}_anno']:
                item_list = data[key]
                num_pad = max_len - len(item_list)
                if num_pad > 0:
                    first_item = item_list[0]
                    pad_item = torch.zeros((num_pad, *first_item.shape), dtype=first_item.dtype)
                    data[key] = torch.cat([torch.stack(item_list, dim=0), pad_item], dim=0)
                else:
                    data[key] = torch.stack(item_list, dim=0)
        
        # 4. Process and stack the raw search images for the APN
        if 'search_images_raw' in data:
            raw_list = data['search_images_raw']
            # Resize and convert to tensor
            search_sz = self.output_sz['search']
            processed_raw = [resize(to_tensor(img), [search_sz, search_sz]) for img in raw_list]

            # Pad the sequence
            num_pad_raw = self.max_len_s - len(processed_raw)
            if num_pad_raw > 0:
                pad_raw = torch.zeros_like(processed_raw[0]).unsqueeze(0).repeat(num_pad_raw, 1, 1, 1)
                data['search_images_raw'] = torch.cat([torch.stack(processed_raw), pad_raw], dim=0)
            else:
                data['search_images_raw'] = torch.stack(processed_raw)

        data['valid'] = True
        return data
