import gradio as gr
import numpy as np
from PIL import Image, ImageOps

def process_image(img, mask):
    # Convert mask to numpy array (if not already)
    mask_array = np.array(mask)
    
    # Since we're working with a binary mask from Gradio's interface,
    # we don't need to manipulate the mask further for this demo.
    # However, note that the mask is a downscaled version based on the 
    # `image_resize_height` and `image_resize_width` parameters in the 
    # `Gradio.Image` component. For precise work, consider processing 
    # at original resolutions or adjusting these parameters.
    
    # Output the mask as is for demonstration; in real scenarios, 
    # you might apply this mask to the original image for segmentation effects.
    return mask

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Paint(label="Draw Mask", image_shape=(224, 224), 
                 brush_sizes=[3], show_labels=False, 
                 canvas_color="#FFFFFF")
    ],
    outputs=gr.Image(type="pil", label="Output Mask"),
    title="Image Mask Drawing Demo",
    description="Upload an image, draw a mask, and see the output.",
)

if __name__ == "__main__":
    demo.launch()