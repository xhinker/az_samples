#%% 
import sys

leffa_repo_path = "/home/andrewzhu/storage_1t_1/github_repos/Leffa"
sys.path.append(leffa_repo_path)

import numpy as np
from PIL import Image

from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from utils.garment_agnostic_mask_predictor import AutoMasker
from utils.densepose_predictor import DensePosePredictor
from utils.utils import resize_and_center

#%%
class LeffaPredictor(object):
    def __init__(self):
        self.densepose_path             = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/densepose"
        self.schp_path                  = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/schp"
        self.densepose_config_path      = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/densepose/densepose_rcnn_R_50_FPN_s1x.yaml"
        self.densepose_weight_path      = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/densepose/model_final_162be9.pkl"
        
        self.sd15_inpaint_model_path    = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/stable-diffusion-inpainting"
        self.leffa_vt_model_path        = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/virtual_tryon.pth"
        self.leffa_vt_model_dc_path     = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/virtual_tryon_dc.pth"
        
        self.sdxl_inpaint_model_path    = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/stable-diffusion-xl-1.0-inpainting-0.1"
        self.leffa_pt_model_path        = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/pose_transfer.pth"
        
        self.mask_predictor = AutoMasker(
            # densepose_path  = "./ckpts/densepose",
            densepose_path  = self.densepose_path,
            # schp_path       = "./ckpts/schp",
            schp_path       = self.schp_path,
        )

        self.densepose_predictor = DensePosePredictor(
            # config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            config_path     = self.densepose_config_path,
            # weights_path="./ckpts/densepose/model_final_162be9.pkl",
            weights_path    = self.densepose_weight_path,
        )

        # virtual tryon still using sd15
        vt_model = LeffaModel(
            # pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model_name_or_path   = self.sd15_inpaint_model_path,
            # pretrained_model="./ckpts/virtual_tryon.pth",
            pretrained_model                = self.leffa_vt_model_path,
        )
        self.vt_inference   = LeffaInference(model=vt_model)
        self.vt_model_type  = "viton_hd"

        # # position transfer using sdxl inpaint
        # pt_model = LeffaModel(
        #     # pretrained_model_name_or_path   = "./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
        #     pretrained_model_name_or_path   = self.sdxl_inpaint_model_path,
        #     # pretrained_model                = "./ckpts/pose_transfer.pth",
        #     pretrained_model                = self.leffa_pt_model_path,
        # )
        # self.pt_inference = LeffaInference(model=pt_model)

    def change_vt_model(self, vt_model_type):
        if vt_model_type == self.vt_model_type:
            return
        if vt_model_type == "viton_hd":
            # pretrained_model = "./ckpts/virtual_tryon.pth"
            pretrained_model = self.leffa_vt_model_path
        elif vt_model_type == "dress_code":
            # pretrained_model = "./ckpts/virtual_tryon_dc.pth"
            pretrained_model = self.leffa_vt_model_dc_path
        vt_model = LeffaModel(
            # pretrained_model_name_or_path = "./ckpts/stable-diffusion-inpainting",
            pretrained_model_name_or_path   = self.sd15_inpaint_model_path,
            pretrained_model                = pretrained_model,
        )
        self.vt_inference   = LeffaInference(model=vt_model)
        self.vt_model_type  = vt_model_type

    def leffa_predict(
        self
        , src_image_path
        , ref_image_path
        , control_type:str  = "virtual_tryon"
        , step:int          = 50
        , scale:float       = 2.5
        , seed:int          = 42
    ):
        assert control_type in ["virtual_tryon", "pose_transfer"], "Invalid control type: {}".format(control_type)
        src_image = Image.open(src_image_path)
        ref_image = Image.open(ref_image_path)
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Mask
        if control_type == "virtual_tryon":
            src_image   = src_image.convert("RGB")
            mask        = self.mask_predictor(src_image, "upper")["mask"]
        elif control_type == "pose_transfer":
            mask        = Image.fromarray(np.ones_like(src_image_array) * 255)

        # DensePose
        if control_type == "virtual_tryon":
            src_image_seg_array = self.densepose_predictor.predict_seg(
                src_image_array
            )
            src_image_seg = Image.fromarray(src_image_seg_array)
            densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(
                src_image_array)
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        # Leffa
        transform = LeffaTransform()

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if control_type == "virtual_tryon":
            inference = self.vt_inference
        elif control_type == "pose_transfer":
            inference = self.pt_inference
        output = inference(
            data
            , num_inference_steps   = step
            , guidance_scale        = scale
            , seed                  = seed
        )
        gen_image = output["generated_image"][0]
        # gen_image.save("gen_image.png")
        # return np.array(gen_image)
        return gen_image

    def leffa_predict_vt(self, src_image_path, ref_image_path, step, scale, seed):
        return self.leffa_predict(src_image_path, ref_image_path, "virtual_tryon", step, scale, seed)

    def leffa_predict_pt(self, src_image_path, ref_image_path, step, scale, seed):
        return self.leffa_predict(src_image_path, ref_image_path, "pose_transfer", step, scale, seed)
    
#%%
leffa_pipe = LeffaPredictor()

#%%
# src_image = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/examples/person1/01376_00.jpg"
# src_image = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/examples/person2/01875_00.jpg"
src_image = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/examples/person2/02902_00.jpg"

ref_image = "/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/franciszzj/Leffa_main/examples/garment/01486_00.jpg"

result = leffa_pipe.leffa_predict_vt(
    src_image_path      = src_image
    , ref_image_path    = ref_image
    , step              = 40
    , scale             = 2.5
    , seed              = 42
)
result
