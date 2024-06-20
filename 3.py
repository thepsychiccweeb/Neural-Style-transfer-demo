import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

class NSTModel(nn.Module):
    def __init__(self):
        super(NSTModel, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

from PIL import Image
import numpy as np

def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    
    if max(np.array(image).shape) > max_size:
        size = max_size
    else:
        size = max(np.array(image).shape)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
        transforms.Resize((size, int(size * image.size[0] / image.size[1]))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    
    return image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def style_transfer(content, style, model, content_weight=1e4, style_weight=1e2, steps=2000):
    target = content.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.003)
    
    style_features = model(style)
    content_features = model(content)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for i in range(steps):
        target_features = model(target)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        style_loss = 0
        for layer in style_features:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (target_feature.shape[1] * target_feature.shape[2])
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return target

import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NSTModel().to(device)

st.title("Neural Style Transfer with Streamlit")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_image = load_image(content_file).to(device)
    style_image = load_image(style_file, shape=content_image.shape[-2:]).to(device)
    
    st.image(im_convert(content_image), caption="Content Image", use_column_width=True)
    st.image(im_convert(style_image), caption="Style Image", use_column_width=True)
    
    if st.button("Generate Style Transfer"):
        output = style_transfer(content_image, style_image, model)
        st.image(im_convert(output), caption="Output Image", use_column_width=True)
