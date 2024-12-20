#%%
import os
import torch
from omegaconf import OmegaConf
import yaml
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
import tqdm
from torch.utils.data._utils.collate import default_collate
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F

os.environ['OMP_NUM_THREADS']           = '1'
os.environ['OPENBLAS_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']           = '1'
os.environ['VECLIB_MAXIMUM_THREADS']    = '1'
os.environ['NUMEXPR_NUM_THREADS']       = '1'

checkpoint_path     = "/home/andrewzhu/storage_1t_1/github_repos/lama/big-lama/models/best.ckpt"
# train_config_path   = "/home/andrewzhu/storage_1t_1/github_repos/lama/big-lama/config.yaml"
train_config_path   = "configs/train_config.yaml"
# predict_config_path = "/home/andrewzhu/storage_1t_1/github_repos/lama/configs/prediction/default.yaml"
predict_config_path = "configs/predict_config.yaml"
device              = "cuda:1"

with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))
    
train_config.training_model.predict_only    = True
train_config.visualizer.kind                = 'noop'

with open(predict_config_path, 'r') as f:
    predict_config = OmegaConf.create(yaml.safe_load(f))

out_ext = predict_config.get('out_ext', '.png')

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')

def load_image_mask_from_uri(image_uri, mask_uri):
    """
    Load image and mask from URIs and create a batch dictionary.
    
    Args:
    - image_uri (str): URI to the image file.
    - mask_uri (str): URI to the mask file.
    
    Returns:
    - batch (dict): Dictionary containing 'image', 'mask', and 'unpad_to_size' (which will be None since no resizing is done).
    """

    # Load image and mask
    image = Image.open(image_uri).convert('RGB')  # Assuming RGB for images
    mask = Image.open(mask_uri).convert('L')      # Grayscale for masks
    
    # Check if image and mask are the same size
    assert image.size == mask.size, "Image and mask must be the same size"
    
    # Convert to tensors
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)/255  # RGB to (C, H, W)
    mask_tensor = torch.from_numpy(np.array(mask))[None, :]/255  # Grayscale to (1, H, W)
    
    # Since target_size matches the image/mask size, no resizing is needed
    unpad_to_size = [torch.tensor(image.size[1]),torch.tensor(image.size[0])]
    
    # Construct batch dictionary
    batch = {
        'image': image_tensor.unsqueeze(0),  # Add batch dimension
        'mask': mask_tensor.unsqueeze(0),    # Add batch dimension
    }
    
    batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
    
    if unpad_to_size:
        batch['unpad_to_size'] = unpad_to_size
    
    return batch

#%% load model and test data
model = load_checkpoint(
    train_config
    , checkpoint_path
    , strict        = False
    , map_location  = 'cpu'
)
model.freeze()
model.to(device)

#%% get image from folder and use make_default_val_dataset 
# if not predict_config.indir.endswith('/'):
#     predict_config.indir += '/'
    
# dataset = make_default_val_dataset(
#     predict_config.indir
#     , **predict_config.dataset
# )

# batch = default_collate([dataset[0]])

#%% load image from uri directly
# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/lama_inpaint/test_images/6458524847_2f4c361183_k.png"
# mask_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/lama_inpaint/test_images/6458524847_2f4c361183_k_mask.png"

# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/image_w_watermark.png"
# mask_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/image_w_watermark_mask.png"

# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/image_w_watermark2.png"
# mask_path  = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/image_w_watermark2_mask.png"

# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/source_images/jeans.png"
# mask_path  = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sematic_seg/output_images/jeans_mask.png"

image_path      = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/lama_inpaint/test_images/image_w_watermark.png"
mask_path       = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/ai_models_eval/lama_inpaint/test_images/image_w_watermark_mask.png"

batch = load_image_mask_from_uri(image_uri=image_path, mask_uri=mask_path)

#%%
print(batch)

#%%
with torch.no_grad():
    batch           = move_to_device(batch, device)
    # ensure all mask are binary number
    batch['mask']   = (batch['mask'] > 0) * 1
    batch           = model(batch)
    
cur_res         = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
unpad_to_size   = batch.get('unpad_to_size', None)
if unpad_to_size is not None:
    orig_height, orig_width = unpad_to_size
    cur_res                 = cur_res[:orig_height, :orig_width]

cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

# output_file_name = "output/result.png"
# cv2.imwrite(output_file_name, cur_res)

#%%
from azailib.image_tools import convert_cv2_to_pil_img
output_img = convert_cv2_to_pil_img(cur_res)
output_img
    