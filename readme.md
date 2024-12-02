# Andrew Zhu's Sample Code

<a href="https://www.amazon.com/Using-Stable-Diffusion-Python-Generation/dp/1835086373" target="_blank"><img src="https://m.media-amazon.com/images/I/81qJBJlgGEL._SL1500_.jpg" alt="Using Stable Diffusion with Python" height="256px" align="right"></a>

Previously, I put my sample code in one of my private repository, the repo becomes like a chaos and impossible to manage, and becomes unreadable (not mention execution it again) gradually. 

So that decide to setup this public repository to host all of my daily sample code with better structure and comments. 

More sample code in my book. 

## Setup Environment

If you want to run some of the code from this repo, to avoid messing up your Python runtime, you'd better setup a virtual python environment and install the packages list in `requirements.txt` file.

Install `azailib` tool

```sh
pip install -U git+https://github.com/xhinker/azailib.git@main
```

Install `sd_embed` for unlimited size Diffusion prompt embedding generation

```sh
pip install -U git+https://github.com/xhinker/sd_embed.git@main
```


## What sample code inside of this rep

### Diffusion related model

1. Stable Diffusion 1.5, see code in folder: `diffusion_models/sd15`
2. Stable Diffusion XL, see code in folder: `diffusion_models/sdxl`
3. Stable Diffusion SD35, see code in folder: `diffusion_models/sd35`
4. Flux.1, see code in folder: `diffusion_models/flux`

### Segmentation

1. Florence2, see code in folder: `segmentations/florence2`
2. GroundingDino: see code in folder: `segatmentations/grounding_dino`

### Image Editing

1. Remove Object & Waters from image - LaMa Inpaint: `ai_models_eval/lama_inpaint`