# In lib/train/actors/odtrack_actor.py

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import os
import torchvision
from torchvision.transforms.functional import crop, resize
from torchvision.utils import draw_bounding_boxes
from pathlib import Path
import matplotlib.cm as cm
import numpy as np
from ...utils.ce_utils import generate_mask_cond


class ODTrackActor(BaseActor):
    """ Actor for training ODTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.iter = 0
        self.viz_interval = 1875

        if settings.local_rank in [-1, 0]:
            self.viz_dir = Path(self.settings.env.workspace_dir) / "viz_apn"
            self.viz_dir.mkdir(exist_ok=True)
            self.mean = torch.tensor(cfg.DATA.MEAN).view(1, 3, 1, 1).to(self.settings.device)
            self.std = torch.tensor(cfg.DATA.STD).view(1, 3, 1, 1).to(self.settings.device)

    def __call__(self, data):
        self.iter += 1
        run_viz = (self.iter % self.viz_interval == 0) and (self.settings.local_rank in [-1, 0])

        out_dict = self.forward_pass(data, run_viz)
        loss, status = self.compute_losses(out_dict, data)
        if run_viz:
            self.save_visualization(out_dict, data)
        return loss, status

    def forward_pass(self, data, run_viz=False):
        num_template_frames = data['template_images'].shape[1]
        num_search_frames = data['search_images'].shape[1]

        template_list = [data['template_images'][:, i, ...] for i in range(num_template_frames)]
        search_list = [data['search_images'][:, i, ...] for i in range(num_search_frames)]
        search_list_raw = [data['search_images_raw'][:, i, ...] for i in range(num_search_frames)]
        
        template_anno_list = [data['template_anno'][:, i, ...] for i in range(num_template_frames)]

        out_dict = self.net(template=template_list,
                            search=search_list,
                            template_history=template_list,
                            run_apn_viz=run_viz,
                            search_images_raw=search_list_raw,
                            template_anno_history=template_anno_list
                            )
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float, device=gt_dict['search_images'][0,0].device)
        
        for i in range(len(pred_dict)):
            gt_bbox = gt_dict['search_anno'][:, i, ...]
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
            
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss
            loss_dict['l1'] = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
            total_loss += loss
            
            if return_status:
                status = {
                    f"{i}frame_Loss/total": loss.item(),
                    f"{i}frame_Loss/giou": loss_dict.get('giou', torch.tensor(0.0)).item(),
                    f"{i}frame_Loss/l1": loss_dict.get('l1', torch.tensor(0.0)).item(),
                    f"{i}frame_IoU": iou.detach().mean().item(),
                }
                if 'apn_confidence' in pred_dict[i]:
                    status[f"{i}frame_APN_Confidence"] = pred_dict[i]['apn_confidence'].mean().item()
                total_status.update(status)
        
        if return_status:
            return total_loss, total_status
        else:
            return total_loss
            
    def tensor_to_colormap(self, tensor, cmap_name='viridis'):
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        np_array = tensor.squeeze().cpu().numpy()
        cmap = cm.get_cmap(cmap_name)
        colored_array = cmap(np_array)[:, :, :3]
        return torch.from_numpy(colored_array).permute(2, 0, 1)

    def save_visualization(self, pred_dict, gt_dict):
        i = 0
        b = 0

        # --- START OF MODIFICATION: Remove check for 'template_mask' ---
        if 'predicted_template' not in pred_dict[i] or 'upsampled_mask' not in pred_dict[i] or 'templates_used' not in pred_dict[i]:
        # --- END OF MODIFICATION ---
            return

        viz_size = (224, 224)
        
        viz_grid = []
        templates_used_norm = pred_dict[i]['templates_used']
        num_gt_templates = gt_dict['template_images'].shape[1]

        for t_idx, t_norm in enumerate(templates_used_norm):
            img = ((t_norm[b].unsqueeze(0) * self.std) + self.mean).clamp(0, 1).cpu().squeeze(0)
            img_resized = resize(img, viz_size)
            
            img_uint8 = (img_resized * 255).to(torch.uint8)
            
            if t_idx < num_gt_templates:
                gt_anno_norm = gt_dict['template_anno'][b, t_idx, :]
                box_xyxy = box_xywh_to_xyxy(gt_anno_norm).cpu()
                box_unnorm = box_xyxy * torch.tensor([viz_size[1], viz_size[0], viz_size[1], viz_size[0]], dtype=torch.float32)
                img_with_box = draw_bounding_boxes(img_uint8, box_unnorm.unsqueeze(0), colors="red", width=2)
            else:
                full_frame_box = torch.tensor([0, 0, viz_size[1]-1, viz_size[0]-1], dtype=torch.float32).unsqueeze(0)
                img_with_box = draw_bounding_boxes(img_uint8, full_frame_box, colors="green", width=2)
            
            viz_grid.append(img_with_box.to(torch.float32) / 255.0)
        
        # --- START OF MODIFICATION: Remove the 'template_mask' visualization logic ---
        # The following block is now removed.
        # template_mask = pred_dict[i]['template_mask'][b].cpu()
        # template_mask_heatmap = self.tensor_to_colormap(template_mask, cmap_name='inferno')
        # template_with_mask = (first_gt_template_img * 0.4 + template_mask_heatmap * 0.6)
        # viz_grid.append(resize(template_with_mask, viz_size))
        # --- END OF MODIFICATION ---
        
        search_frame_norm = gt_dict['search_images_raw'][b, i, ...].unsqueeze(0)
        search_frame = search_frame_norm.clamp(0, 1).cpu()
        
        predicted_template_norm = pred_dict[i]['predicted_template'][b].unsqueeze(0)
        predicted_template = predicted_template_norm.clamp(0, 1).cpu()

        upsampled_mask = pred_dict[i]['upsampled_mask'][b].unsqueeze(0)
        mask_heatmap = self.tensor_to_colormap(upsampled_mask)
        search_frame_with_mask = (search_frame.squeeze(0) * 0.4 + mask_heatmap * 0.6).unsqueeze(0)

        gt_bbox_xywh = gt_dict['search_anno'][b, i, ...]
        search_image_gt_processed = gt_dict['search_images'][b, i, ...].unsqueeze(0)
        search_image_gt_denorm = ((search_image_gt_processed * self.std) + self.mean).clamp(0, 1).cpu()
        
        img_h, img_w = search_image_gt_denorm.shape[-2:]
        gt_bbox_abs = box_xywh_to_xyxy(gt_bbox_xywh.cpu()) * torch.tensor([img_w, img_h, img_w, img_h])
        
        x1, y1, x2, y2 = gt_bbox_abs[0], gt_bbox_abs[1], gt_bbox_abs[2], gt_bbox_abs[3]
        gt_crop = crop(search_image_gt_denorm.squeeze(0), int(y1), int(x1), int(y2 - y1), int(x2 - x1))
        
        num_templates = len(templates_used_norm)
        max_templates_cfg = self.cfg.DATA.TEMPLATE.get("MAX_TOTAL_TEMPLATES", 3)
        if num_templates < max_templates_cfg:
            viz_grid.extend([torch.ones(3, *viz_size)] * (max_templates_cfg - num_templates))

        main_viz_components = [search_frame, search_frame_with_mask, predicted_template, gt_crop.unsqueeze(0)]
        viz_grid.extend([resize(img.squeeze(0), viz_size) for img in main_viz_components])

        confidence = pred_dict[i]['apn_confidence'][b].item()
        filepath = self.viz_dir / f"epoch_{gt_dict['epoch']}_iter_{self.iter}_conf_{confidence:.3f}.png"
        
        # Adjust nrow for the removed template mask image
        torchvision.utils.save_image(torch.stack(viz_grid), filepath, nrow=max_templates_cfg + 4)