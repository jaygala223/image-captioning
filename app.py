import streamlit as st
from PIL import Image
import tensorflow as tf
from torch_code import *
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title("Image Captioning")

input_img = st.file_uploader("Upload image to generate captions", accept_multiple_files=False,key="1")

if input_img:
    st.image(input_img, width=250)

generate_captions_buttons = st.button("Generate captions", key='generate_captions')

if generate_captions_buttons and input_img:
    #do something
    model = VisionEncoderDecoderModel.from_pretrained("model_pretrained")
    feature_extractor = ViTFeatureExtractor.from_pretrained("feature_extractor_pretrained")
    tokenizer = AutoTokenizer.from_pretrained("tokenizer_pretrained")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    def predict_step(image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
        
    st.write("Caption: ", predict_step([input_img])[0])
    
