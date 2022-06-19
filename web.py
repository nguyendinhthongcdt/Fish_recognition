# Đoạn code này không cần quan tâm có chạy được trên python hay không, vì nó sẽ chạy trực tiếp trên github, do trên python chưa cài streamlit
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("Hue.h5") #model m train

### load file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])

map_dict = {0: 'Cá ba đuôi',
            1: 'Cá cánh buồm hồng',
            2:'Cá chép sư tử trắng',
            3:'Cá hạc đỉnh hồng',
            4:'Cá hổ bạc',
            5:'Cá la hán',
            6:'Cá mã giáp hoàng kim',
            7:'Cá mún panda',
            8:'Cá neon',
            9:'Cá phượng hoàng',
            10:'Cá rồng huyết long',
            11:'Cá rồng kim long',
            12:'Cá sam black diamond',
            13:'Cá thần tiên',
            14:'Cá tứ vân'}
    
 
if uploaded_file is not None:
    # Convert the file
    img = image.load_img(uploaded_file,target_size=(64,64)) #xử lí ảnh theo cách m làm
    st.image(uploaded_file, channels="RGB") #hiển thị ảnh
    img = img_to_array(img)
    img = img.reshape(1,64,64,3)
    img = img.astype('float32')
    img = img/255
        
    #Button: nút dự đoán sau khi up ảnh
    Genrate_pred = st.button("Generate Prediction") 
    
    if Genrate_pred:
    
        prediction = model.predict(img).argmax()
        st.write("**Predicted Label for the image is {}**".format(map_dict [prediction])) ##đưa ra dự đoán viêm phổi hay ko
