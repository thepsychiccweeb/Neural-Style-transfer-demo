import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    model_outputs = content_outputs + style_outputs
    return Model(vgg.input, model_outputs), style_layers

def get_feature_representations(model, content_path, style_path):
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    content_features = [content_layer[0] for content_layer in content_outputs[:1]]
    style_features = [style_layer[0] for style_layer in style_outputs[1:]]
    return content_features, style_features

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

from PIL import Image
import numpy as np
import tensorflow as tf

def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.LANCZOS)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3 
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocess image")
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features, style_layers):
    model_outputs = model(init_image)
    style_output_features = model_outputs[1:]
    content_output_features = model_outputs[:1]
    style_score = 0
    content_score = 0
    weight_per_style_layer = 1.0 / float(len(style_layers))
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * tf.reduce_mean(tf.square(comb_style - target_style))
    content_score += tf.reduce_mean(tf.square(content_output_features[0] - content_features[0]))
    style_score *= loss_weights[1]
    content_score *= loss_weights[0]
    loss = style_score + content_score
    return loss, style_score, content_score

def compute_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

def style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2): 
    model, style_layers = get_model()
    for layer in model.layers:
        layer.trainable = False
    content_features, style_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    init_image = load_and_process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=1e-1)  # corrected learning rate to 5.0
    iter_count = 1
    best_loss, best_img = float('inf'), None
    loss_weights = (content_weight, style_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features,
        'style_layers': style_layers
    }
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
        if i % 100 == 0:
            print('Iteration: {}'.format(i))        
    return best_img

import streamlit as st

st.title("Neural Style Transfer with Streamlit")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_path = content_file.name
    style_path = style_file.name
    
    with open(content_path, "wb") as f:
        f.write(content_file.getbuffer())
        
    with open(style_path, "wb") as f:
        f.write(style_file.getbuffer())

    st.image(load_img(content_path), caption="Content Image", use_column_width=True)
    st.image(load_img(style_path), caption="Style Image", use_column_width=True)
    
    if st.button("Generate Style Transfer"):
        output = style_transfer(content_path, style_path)
        st.image(output, caption="Output Image", use_column_width=True)
