# In lib/train/actors/odtrack_actor.py

from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
# --- START OF MODIFICATION ---
# These imports are no longer needed for loss calculation here
# from torchvision.transforms.functional import crop, resize
# --- END OF MODIFICATION ---
# --- START OF VISUALIZATION MODIFICATION ---
import os
import torchvision
from torchvision.transforms.functional import crop, resize
from pathlib import Path
# --- END OF VISUALIZATION MODIFICATION ---
class ODTrackActor(BaseActor):
    """ Actor for training ODTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.iter = 0 # Iteration counter for periodic saving
        self.viz_interval = 50 # Save a visualization every 200 iterations
        if settings.local_rank != -1:
            self.viz_dir = Path(self.settings.env.workspace_dir) / "viz_apn"
            self.viz_dir.mkdir(exist_ok=True)
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
        # --- START OF VISUALIZATION MODIFICATION ---
        self.iter += 1
        run_viz = (self.iter % self.viz_interval == 0) and (self.settings.local_rank in [-1, 0])
        # --- END OF VISUALIZATION MODIFICATION ---
        # forward pass
        out_dict = self.forward_pass(data,run_viz)
        
        # compute losses
        loss, status = self.compute_losses(out_dict, data)
        if run_viz:
            self.save_visualization(out_dict, data)
        return loss, status

    def forward_pass(self, data,run_viz=False):
        # The ODTrack model expects a list of templates.
        # The number of historical templates is defined by cfg.DATA.TEMPLATE.NUM_HISTORY
        template_list = [data['template_images'][i].view(-1, *data['template_images'].shape[2:])
                         for i in range(self.settings.num_template)]

        search_list = [data['search_images'][i].view(-1, *data['search_images'].shape[2:])
                       for i in range(self.settings.num_search)]
            
        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = [generate_mask_cond(self.cfg, img.shape[0], img.device, anno)
                          for img, anno in zip(template_list, data['template_anno'])]
            box_mask_z = torch.cat(box_mask_z, dim=1) if box_mask_z else None

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # --- START OF MODIFICATION ---
        # Pass the historical templates to the network. The network itself will decide if/how to use the APN.
        out_dict = self.net(template=template_list,
                            search=search_list,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            template_history=template_list,
                            # --- START OF VISUALIZATION MODIFICATION ---
                            run_apn_viz=run_viz)
                            # --- END OF VISUALIZATION MODIFICATION ---
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
            # Appearance prediction loss is no longer computed. The APN is frozen and not trained.
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
                total_status.update(status)
        
        if return_status:
            return total_loss, total_status
        else:
            return total_loss
     # --- START OF VISUALIZATION MODIFICATION ---
    def save_visualization(self, pred_dict, gt_dict):
        """Saves a grid of images for debugging the APN."""
        i = 0
        if 'predicted_template' not in pred_dict[i] or 'upsampled_mask' not in pred_dict[i]:

            return

        # --- START OF FIX ---
        # Define a common size for visualization
        viz_size = (384, 384)

        # Get all tensors and move them to CPU
        input_template = gt_dict['template_images'][-1][i].cpu()
        search_frame = gt_dict['search_images'][0][i].cpu()
        predicted_template = pred_dict[i]['predicted_template'][i].cpu()
        upsampled_mask = pred_dict[i]['upsampled_mask'][i].cpu()

        # Resize all tensors to the common visualization size
        input_template_resized = resize(input_template, viz_size)
        search_frame_resized = resize(search_frame, viz_size)
        predicted_template_resized = resize(predicted_template, viz_size)
        
        # The mask also needs resizing and conversion to 3-channel for stacking
        mask_viz_resized = resize(upsampled_mask, viz_size).repeat(3, 1, 1)

        # Generate and resize the ground-truth template
        search_image_gt = gt_dict['search_images'][i]
        gt_bbox_xywh = gt_dict['search_anno'][i]
        img_h, img_w = search_image_gt.shape[-2:]
        gt_bbox_abs = box_xywh_to_xyxy(gt_bbox_xywh[i]) * torch.tensor([img_w, img_h, img_w, img_h], device=search_image_gt.device)
        gt_crop = crop(search_image_gt[i], int(gt_bbox_abs[1]), int(gt_bbox_abs[0]), int(gt_bbox_abs[3]-gt_bbox_abs[1]), int(gt_bbox_abs[2]-gt_bbox_abs[0]))
        gt_template_resized = resize(gt_crop, viz_size).cpu()

        # Create the grid using the resized tensors
        viz_grid = torch.stack([
            input_template_resized,
            search_frame_resized,
            mask_viz_resized,
            predicted_template_resized,
            gt_template_resized
        ])
        # --- END OF FIX ---

        filepath = self.viz_dir / f"epoch_{gt_dict['epoch']}_iter_{self.iter}.png"
        search_frame_path = self.viz_dir / f"search_for_epoch_{gt_dict['epoch']}_iter_{self.iter}.png"
        mask_frame_path = self.viz_dir / f"mask_for_epoch_{gt_dict['epoch']}_iter_{self.iter}.png"
        torchvision.utils.save_image(viz_grid, filepath, nrow=5, normalize=True, value_range=(0, 1))
        torchvision.utils.save_image(upsampled_mask, mask_frame_path, nrow=1, normalize=True, value_range=(0, 1))
        torchvision.utils.save_image(search_frame, search_frame_path, nrow=1, normalize=True, value_range=(0, 1))