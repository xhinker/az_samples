#%%
from segmentations.sematic_seg.common_tools import *

checkpoint = "/home/andrewzhu/storage_1t_1/github_repos/sam2/checkpoints/sam2.1_hiera_large.pt"
# the following format is required, the original writer did it wrong for full absolute path
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda:0"

#%% load up model
# any benefit of doing this?
# torch.autocast(device, dtype=torch.bfloat16).__enter__()

predictor = load_sam2_model(
    checkpoint_path     = checkpoint
    , model_cfg_path    = model_cfg
    , device            = device
)

#%%
# image_path  = "one_truck.png"
image_path  = "images/jeans.png"
image       = load_image(image_path)
predictor.set_image(image)

#%% select object with point
input_point = np.array([[500, 375]])
input_label = np.array([1])

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

# print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

#%% predict one area
masks, scores, logits = predictor.predict(
    point_coords        = input_point
    , point_labels      = input_label
    , multimask_output  = False
)
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)


#%%
#%% predict multiple areas
masks, scores, logits = predictor.predict(
    point_coords        = input_point
    , point_labels      = input_label
    , multimask_output  = True
)

sorted_ind  = np.argsort(scores)[::-1]
masks       = masks[sorted_ind]
scores      = scores[sorted_ind]
logits      = logits[sorted_ind]

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

#%% select object with box, do not support multiple boxes
# input_box = np.array([425, 600, 700, 875])
input_box = np.array([401, 0, 695, 387])

masks, scores, _ = predictor.predict(
    point_coords    = None,
    point_labels    = None,
    box             = input_box[None, :],
    multimask_output=False,
)
show_masks(image, masks, scores, box_coords=input_box)

#%%
pil_mask = get_mask_img(masks[0])
pil_mask

#%%
# # import cv2
# # mask = masks[0]
# # print(mask.shape)
# # h, w = mask.shape[-2:]
# # #mask = mask.astype(np.uint8)

# # # color = np.array([30/255, 144/255, 255/255, 0.6])
# # # color = np.array([30/255, 144/255, 255/255])
# # color = np.array([1, 1, 1])
# # # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
# # mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, 3)
# # mask_image.shape

# # mask_image_uint8 = ((1 - mask_image) * 255).astype(np.uint8) 
# # img_rgb = cv2.cvtColor(mask_image_uint8, cv2.COLOR_BGR2RGB)
# # pil_img = Image.fromarray(img_rgb)
# # pil_img

# #%%
# # plt.figure(figsize=(10, 10))
# # plt.imshow(image)
# # mask = show_mask(masks[0], plt.gca())


# #plt.show()
# #mask
# #%%
# mask.shape[-2:]
# # import cv2
# # img_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
# # img_rgb.shape