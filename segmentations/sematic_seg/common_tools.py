import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from diffusers.utils import load_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from groundingdino.util import box_ops
from torchvision.ops import box_convert

def get_xyxy_boxes(cxcywh_boxes, image_source:Image):
    '''
    convert the float cxcywh boxes to int xyxy boxes
    '''
    h, w, _ = image_source.shape
    xyxy_boxes_output = []
    boxes = cxcywh_boxes * torch.Tensor([w,h,w,h])
    xyxy_boxes = box_convert(
        boxes       = boxes
        , in_fmt    = "cxcywh"
        , out_fmt   = "xyxy"
    )
    for xyxy_box in xyxy_boxes:
        xyxy_box = [int(i) for i in xyxy_box]
        xyxy_boxes_output.append(xyxy_box)
    return xyxy_boxes_output

def show_cv2_img(image_data):
    img_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    display(pil_img)

def load_sam2_model(
    checkpoint_path:str
    , model_cfg_path:str
    , device:str = "cuda:0"
):
    predictor = SAM2ImagePredictor(
        build_sam2(
            model_cfg_path
            , checkpoint_path
            , device = device
        )
    )
    return predictor

def segment(
    sam_model
    , image_path:str
    , input_box:np.ndarray
):
    image = load_image(image_path)
    sam_model.set_image(image)
    
    masks, scores, _ = sam_model.predict(
        point_coords    = None,
        point_labels    = None,
        box             = input_box[None, :],
        multimask_output=False,
    )
    return masks.cpu()

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray(
        (mask_image.cpu().numpy() * 255).astype(np.uint8)
    ).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    return mask_image

def get_mask_img(mask:np.ndarray):
    # get h and w, -2 means get last two numbers of shape array
    h, w = mask.shape[-2:]
    # set white background color
    color = np.array([1, 1, 1])
    # expand mask image to hxwxc (c is 3 channels here)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, 3)
    # OpenCV primarily works with 8-bit (uint8) or 32-bit floating-point (float32) images for most operations.
    mask_image_uint8 = ((1 - mask_image) * 255).astype(np.uint8) 
    # convert BGR to RGB
    mask_rgb = cv2.cvtColor(mask_image_uint8, cv2.COLOR_BGR2RGB)
    # convert cv2 image to PIL image
    pil_mask = Image.fromarray(mask_rgb)
    return pil_mask

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()