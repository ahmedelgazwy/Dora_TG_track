# In lib/train/actors/odtrack_actor.py

from . import BaseActor
from lib.utils.misc import NestedTensor, interpolate
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import os
import torchvision
from torchvision.transforms.functional import crop, resize
from pathlib import Path


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

    def forward_pass(self, data, run_viz=False):
        # --- START OF FIX ---
        # Data is now nested in 'template' and 'search' keys
        template_data = data['template']
        search_data = data['search']
        
        template_list = template_data['images']
        search_list = search_data['images']
        raw_search_frames = search_data.get('raw_search_frames')
        # --- END OF FIX ---
            
        box_mask_z = None # Not implemented in this simplified flow
        ce_keep_rate = None # Not implemented in this simplified flow

        out_dict = self.net(template=template_list,
                            search=search_list,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            template_history=template_list,
                            run_apn_viz=run_viz,
                            raw_search_frames_for_apn=raw_search_frames)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # The logic here is already correct as it loops through the list of predictions
        # ... (This function remains unchanged)
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float, device=gt_dict['search']['images'][0].device)
        
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search']['anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        
        for i in range(len(pred_dict)):
            gt_bbox = gt_dict['search']['anno'][i]
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

    def save_visualization(self, pred_dict_list, gt_dict):
        """Saves a grid of images for debugging the APN for each search frame."""
        viz_size = (224, 224)
        
        # We visualize for the first item in the batch
        batch_idx = 0

        # Loop through each search frame's prediction
        for i, pred_dict in enumerate(pred_dict_list):
            if 'predicted_template' not in pred_dict or 'upsampled_mask' not in pred_dict:
                continue

            # Get the input template (it's the same for all search frames in this sequence)
            input_template = gt_dict['template']['images'][-1][batch_idx].cpu()
            
            # Get the specific raw search frame for this timestep
            raw_search_frame = gt_dict['search']['raw_search_frames'][batch_idx, i].cpu()
            
            predicted_template = pred_dict['predicted_template'][batch_idx].cpu()
            upsampled_mask = pred_dict['upsampled_mask'][batch_idx].cpu()

            # Resize all components
            input_template_r = resize(input_template, viz_size)
            search_frame_r = resize(raw_search_frame, viz_size)
            predicted_template_r = resize(predicted_template, viz_size)
            mask_viz_r = resize(upsampled_mask, viz_size).repeat(3, 1, 1)

            # Generate GT template for this specific search frame
            gt_bbox_xywh = gt_dict['search']['anno'][i][batch_idx]
            img_h, img_w = raw_search_frame.shape[-2:]
            gt_bbox_abs = box_xywh_to_xyxy(gt_bbox_xywh) * torch.tensor([img_w, img_h, img_w, img_h], device=gt_bbox_xywh.device)
            gt_crop = crop(raw_search_frame, int(gt_bbox_abs[1]), int(gt_bbox_abs[0]), int(gt_bbox_abs[3]-gt_bbox_abs[1]), int(gt_bbox_abs[2]-gt_bbox_abs[0]))
            gt_template_r = resize(gt_crop, viz_size).cpu()
            
            # Create and save the grid for this timestep
            viz_grid = torch.stack([
                input_template_r, search_frame_r, mask_viz_r, predicted_template_r, gt_template_r
            ])
            
            # Add timestep index to filename
            filepath = self.viz_dir / f"epoch_{gt_dict['epoch']}_iter_{self.iter}_frame_{i}.png"
            torchvision.utils.save_image(viz_grid, filepath, nrow=5, normalize=True, value_range=(0, 1))