# Thêm các thư viện cần thiết
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("model.h5") 

st.header("Môn học: Trí tuệ nhân tạo")
st.header("GVHD : PGS.TS Nguyễn Trường Thịnh")
st.header("Họ và tên : Nguyễn Đình Thông")
st.header("MSSV : 19146398")

st.title("Nhận diện các loại cá cảnh")
st.image('bg1.jpg')
# Upload file ảnh
uploaded_file = st.file_uploader("Upload hình cá cảnh", type=["jpg","jpeg","png","bmp"])
 
if uploaded_file is not None:
    # Chỉnh sửa ảnh đầu vào
    img = image.load_img(uploaded_file,target_size=(64,64)) 
    st.image(uploaded_file, channels="RGB") 
    img = img_to_array(img)
    img = img.reshape(1,64,64,3)
    img = img.astype('float32')
    img = img/255
# Gắn nhãn các lớp
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
            
    # Nút nhấn
    Genrate_pred = st.button("Dự đoán") 
    
    if Genrate_pred:
        prediction = model.predict(img).argmax()
        st.write("**Kết quả dự đoán {}**".format(map_dict [prediction])) 

