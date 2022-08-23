import streamlit as st
from detect_gun import DetectGun
import os


detect = DetectGun()

st.header('Gun Detection')
#st.file_uploader("Upload Image", type=None, accept_multiple_files=False, key=None, help=None, on_change=None)
imageLocation = st.empty()
img = st.file_uploader("")
if img:
    if not os.path.exists("tempDir"):
        os.makedirs("tempDir")
    with open(os.path.join("tempDir",img.name),"wb") as f:
        f.write(img.getbuffer())
    file_path = os.path.join("tempDir",img.name)
    imageLocation.image(img)
    #st.image(img)
    im0 = detect.detect_gun(file_path)
    imageLocation.image(im0)


