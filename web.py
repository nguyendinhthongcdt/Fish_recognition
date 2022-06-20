# Thêm các thư viện cần thiết
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

# Tải model
model = tf.keras.models.load_model("model.h5") 
#st.sidebar.['Information']
with st.sidebar:
            st.sidebar.text("Đây là sidebar")
# Phần thông tin
c1,c2=st.columns(2)
with c2:
            st.image('bg1.jpg')
with c1:
            st.text("Môn học: Trí tuệ nhân tạo")
            st.text("GVHD : PGS.TS Nguyễn Trường Thịnh")
            st.text("Họ và tên : Nguyễn Đình Thông 19146398")
st.title("NHẬN DIỆN CÁC LOẠI CÁ CẢNH")

c3,c4,c5=st.columns(3)
with c3:
            st.image("acc.png",caption="acc")
with c4:
            st.image("chart.png",caption="chart")
with c5:
            st.image("loss.png",caption="loss")
            


# Upload file ảnh
uploaded_file = st.file_uploader("UPLOAD HÌNH ẢNH CÁ CẢNH", type=["jpg","jpeg","png","bmp"])

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
    # Chỉnh sửa ảnh đầu vào
    img = image.load_img(uploaded_file,target_size=(64,64)) 
    st.image(uploaded_file, channels="RGB") 
    img = img_to_array(img)
    img = img.reshape(1,64,64,3)
    img = img.astype('float32')
    img = img/255
        
    # Nút nhấn
    Genrate_pred = st.button("Dự đoán") 
            
    # Dự đoán và hiển thị kết quả
    if Genrate_pred:
        prediction = model.predict(img).argmax()
        st.write("**Kết quả nhận diện: {}**".format(map_dict [prediction])) 
st.warning("Lưu ý: Trang web này chỉ mang tính chất thao khảo")

