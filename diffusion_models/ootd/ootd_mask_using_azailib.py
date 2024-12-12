'''
Test the ootd library from azailib
'''
#%%
from azailib.image_model_pipes import OOTDPipe

body_pose_checkpoint_path           = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/openpose/ckpts/body_pose_model.pth"
humanparsing_atr_checkpoint_path    = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/humanparsing/parsing_atr.onnx"
humanparsing_lip_checkpoint_path    = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/checkpoints/humanparsing/parsing_lip.onnx"

ootd_pipe = OOTDPipe(
    body_pose_checkpoint_path           = body_pose_checkpoint_path
    , humanparsing_atr_checkpoint_path  = humanparsing_atr_checkpoint_path
    , humanparsing_lip_checkpoint_path  = humanparsing_lip_checkpoint_path
)

#%%
image_path = "/home/andrewzhu/storage_1t_1/github_repos/OOTDiffusion/run/examples/model/01861_00.jpg"
ootd_pipe.get_mask(
    image_or_path = image_path
    , body_position = "lower_body"
    , dilate_margin = 20
)