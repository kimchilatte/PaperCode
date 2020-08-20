import streamlit as st
import numpy as np
import pandas as pd
import torch

from papers.GAN import Generator
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

ckpts_path = Path("./papers") / "GAN" / "ckpts" / "last.ckpt"
ckpt = torch.load(ckpts_path)
G_weight = OrderedDict({k.lstrip("G."): v for k, v in ckpt["state_dict"].items() if "G." in k})

latent_dim = 5
generator = Generator(latent_dim)
generator.load_state_dict(G_weight)
generator.eval()

z = torch.zeros(latent_dim)
# zz = torch.linspace(-1, 1)

st.title("Paper Demo")
st.markdown("Please use side bar on the left, control the 5 factors to generate different image")

L1 = st.sidebar.slider("Latent 1", min_value=-1., max_value=1., value=0., step=1/100, format="%f")
L2 = st.sidebar.slider("Latent 2", min_value=-1., max_value=1., value=0., step=1/100, format="%f")
L3 = st.sidebar.slider("Latent 3", min_value=-1., max_value=1., value=0., step=1/100, format="%f")
L4 = st.sidebar.slider("Latent 4", min_value=-1., max_value=1., value=0., step=1/100, format="%f")
L5 = st.sidebar.slider("Latent 5", min_value=-1., max_value=1., value=0., step=1/100, format="%f")

z[0] = L1
z[1] = L2
z[2] = L3
z[3] = L4
z[4] = L5

# def generate_img(latent_dim, L1, L2, L3, L4, L5):
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
def generate_img(generator, z, ax):
    output = generator(z).detach().squeeze().numpy()
    ax.imshow(output)

generate_img(generator, z, ax)
st.write(fig)