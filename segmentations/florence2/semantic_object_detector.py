#%% load packages for all
from transformers import (
    AutoProcessor
    , AutoModelForCausalLM  
)
import requests
import copy
from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  

device = "cuda:0"

# prepare florence2 model
def run_florence2(
    processor
    , model
    , image
    , task_prompt
    , text_input    = None
    , device        = "cuda:0"
):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
        
    inputs = processor(
        text                = prompt
        , images            = image
        , return_tensors    ="pt"
    )
    inputs.to(device)
    generated_ids = model.generate(
        input_ids           = inputs["input_ids"],
        pixel_values        = inputs["pixel_values"],
        max_new_tokens      = 1024,
        early_stopping      = False,
        do_sample           = False,
        num_beams           = 3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task        = task_prompt, 
        image_size  = (image.width, image.height)
    )

    return parsed_answer

def convert_to_od_format(data):  
    """  
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
  
    Parameters:  
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.  
  
    Returns:  
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
    """  
    # Extract bounding boxes and labels  
    bboxes = data.get('bboxes', [])  
    labels = data.get('bboxes_labels', [])  
      
    # Construct the output format  
    od_results = {  
        'bboxes': bboxes,  
        'labels': labels  
    }  
      
    return od_results  

def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  
      
    # Display the image  
    ax.imshow(image)  
      
    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  
      
    # Remove the axis ticks and labels  
    ax.axis('off')  
      
    # Show the plot  
    plt.show()

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
def draw_polygons(image, prediction, fill_mask=False):  
    """  
    Draws segmentation masks with polygons on an image.  
  
    Parameters:  
    - image_path: Path to the image file.  
    - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                  'polygons' is a list of lists, each containing vertices of a polygon.  
                  'labels' is a list of labels corresponding to each polygon.  
    - fill_mask: Boolean indicating whether to fill the polygons with color.  
    """  
    # Load the image  
   
    draw = ImageDraw.Draw(image)  
      
   
    # Set up scale factor if needed (use 1 if not scaling)  
    scale = 1  
      
    # Iterate over polygons and labels  
    for polygons, label in zip(prediction['polygons'], prediction['labels']):  
        color = random.choice(colormap)  
        fill_color = random.choice(colormap) if fill_mask else None  
          
        for _polygon in polygons:  
            _polygon = np.array(_polygon).reshape(-1, 2)  
            if len(_polygon) < 3:  
                print('Invalid polygon:', _polygon)  
                continue  
              
            _polygon = (_polygon * scale).reshape(-1).tolist()  
              
            # Draw the polygon  
            if fill_mask:  
                draw.polygon(_polygon, outline=color, fill=fill_color)  
            else:  
                draw.polygon(_polygon, outline=color)  
              
            # Draw the label text  
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  
  
    # Save or display the image  
    #image.show()  # Display the image  
    display(image)


#%%

# florence2_model_path = '/home/andrewzhu/storage_14t_5/ai_models_all/sd_hf_models/microsoft/Florence-2-large_main'
florence2_model_path = '/home/andrewzhu/storage_14t_5/ai_models_all/seg_hf_models/microsoft/Florence-2-large-ft_main'

florence2_processor = AutoProcessor.from_pretrained(
    florence2_model_path
    , trust_remote_code = True
)

florence2_model = AutoModelForCausalLM.from_pretrained(
    florence2_model_path
    , trust_remote_code = True
).eval()

florence2_model.to(device)

#%%
# image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/az_samples/segmentations/sam2/one_truck.png"
image_path = "/home/andrewzhu/storage_1t_1/az_git_folder/azcode/az_projects/model_tests/.model_test/image-2.png"
image = load_image(image_path).convert('RGB')
display(image)

#%% 

#%% '<REFERRING_EXPRESSION_SEGMENTATION>'
task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
text_input =  "white dress" # good
results = run_florence2(
    processor       = florence2_processor
    , model         = florence2_model
    , image         = image
    , task_prompt   = task_prompt
    , text_input    = text_input
    , device        = device  
)
print(results)

output_image = copy.deepcopy(image)
draw_polygons(output_image, results[task_prompt], fill_mask=True)  

#%% '<CAPTION_TO_PHRASE_GROUNDING>'
task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
# text_input =  "shoulder" # good
text_input =  "white dress" # good
# text_input =  "left hand" # good
# text_input =  "right hand" # not good
# text_input =  "hand bag" # not good

results = run_florence2(
    processor       = florence2_processor
    , model         = florence2_model
    , image         = image
    , task_prompt   = task_prompt
    , text_input    = text_input
    , device        = device  
)
print(results)
plot_bbox(image, results[task_prompt])


#%% '<OPEN_VOCABULARY_DETECTION>'
task_prompt = '<OPEN_VOCABULARY_DETECTION>'
# text_input =  "shoulder" # good
# text_input =  "left hand" # good
# text_input =  "right hand" # not good
# text_input =  "hand bag" # not good
text_input =  "white dress" # not good

results = run_florence2(
    processor       = florence2_processor
    , model         = florence2_model
    , image         = image
    , task_prompt   = task_prompt
    , text_input    = text_input
    , device        = device  
)
print(results)
bbox_results  = convert_to_od_format(results[task_prompt])
plot_bbox(image, bbox_results)


#%% '<REGION_TO_SEGMENTATION>' similar to segmentation anything, but not very good
# get the cooridate 
box_loc_array = results[task_prompt]['bboxes'][0]
box_loc_array_int = [int(i) for i in box_loc_array]
task_prompt = '<REGION_TO_SEGMENTATION>'
text_input = ""
for loc_int in box_loc_array_int:
    text_input = text_input + f"<loc_{loc_int}>"
print(text_input)

results = run_florence2(
    processor       = florence2_processor
    , model         = florence2_model
    , image         = image
    , task_prompt   = task_prompt
    , text_input    = text_input
    , device        = device  
)
print(results)
output_image = copy.deepcopy(image)
draw_polygons(output_image, results['<REGION_TO_SEGMENTATION>'], fill_mask=True)  