# In lib/train/actors/odtrack_actor.py

from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
# --- START OF MODIFICATION ---
from torchvision.transforms.functional import crop, resize
# --- END OF MODIFICATION ---

class ODTrackActor(BaseActor):
    """ Actor for training ODTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        template_list = []
        search_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            search_list.append(search_img_i)
            
        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(self.settings.num_template):
                box_mask_z.append(generate_mask_cond(self.cfg, template_list[i].shape[0], template_list[i].device,
                                                    data['template_anno'][i]))
            box_mask_z = torch.cat(box_mask_z, dim=1)

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # --- START OF MODIFICATION ---
        # Pass the historical templates to the network. The network itself will decide if to use the APN.
        out_dict = self.net(template=template_list,
                            search=search_list,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            template_history=template_list) # Pass history to APN
        # --- END OF MODIFICATION ---

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float, device=gt_dict['search_images'][0].device)
        
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        
        for i in range(len(pred_dict)):
            # get GT for tracking loss
            gt_bbox = gt_dict['search_anno'][i]
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get predicted boxes for tracking loss
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
            
            # compute giou and l1 loss for tracking
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss
            loss_dict['l1'] = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            
            # compute location loss for tracking
            if 'score_map' in pred_dict[i]:
                loss_dict['focal'] = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                loss_dict['focal'] = torch.tensor(0.0, device=loss_dict['l1'].device)
                
            # --- START OF MODIFICATION ---
            # Compute appearance prediction loss if APN is used
            if 'predicted_template' in pred_dict[i]:
                predicted_template = pred_dict[i]['predicted_template']
                
                # Generate ground-truth template on-the-fly by cropping the search image
                search_image = gt_dict['search_images'][i]
                gt_bbox_xywh = gt_dict['search_anno'][i] # Box is (x, y, w, h) normalized
                
                # Convert normalized xywh to absolute xyxy for cropping
                img_h, img_w = search_image.shape[-2:]
                gt_bbox_abs = box_xywh_to_xyxy(gt_bbox_xywh) * torch.tensor([img_w, img_h, img_w, img_h], device=search_image.device)
                
                # Crop and resize to create the ground-truth template
                gt_templates_cropped = [crop(img, int(box[1]), int(box[0]), int(box[3]-box[1]), int(box[2]-box[0])) for img, box in zip(search_image, gt_bbox_abs)]
                target_size = predicted_template.shape[-2:]
                gt_template = torch.stack([resize(t, target_size) for t in gt_templates_cropped])

                # Calculate appearance losses
                loss_dict['l1_appearance'] = self.objective['l1_appearance'](predicted_template, gt_template)
                loss_dict['perceptual'] = self.objective['perceptual'](predicted_template, gt_template).mean()
            # --- END OF MODIFICATION ---

            # weighted sum of all losses
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
            total_loss += loss
            
            if return_status:
                status = {
                    f"{i}frame_Loss/total": loss.item(),
                    f"{i}frame_Loss/giou": loss_dict.get('giou', torch.tensor(0.0)).item(),
                    f"{i}frame_Loss/l1": loss_dict.get('l1', torch.tensor(0.0)).item(),
                    f"{i}frame_Loss/location": loss_dict.get('focal', torch.tensor(0.0)).item(),
                    f"{i}frame_IoU": iou.detach().mean().item()
                }
                # Add appearance loss to status log if calculated
                if 'l1_appearance' in loss_dict:
                    status[f"{i}frame_Loss/l1_appearance"] = loss_dict['l1_appearance'].item()
                    status[f"{i}frame_Loss/perceptual"] = loss_dict['perceptual'].item()
                total_status.update(status)
        
        if return_status:
            return total_loss, total_status
        else:
            return total_loss