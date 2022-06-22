# Import các thư viện
from pandas import array
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Tắt các ký tự đặt biệt, để khi thực hiện không hiển thị các ký tự này
np.set_printoptions(suppress=True)

# Tải model
model = tensorflow.keras.models.load_model('model.h5')
# Tạo mảng có kích thước phù hợp để đưa vào model
data = np.ndarray(shape=(3, 64, 64, 3), dtype=np.float32)
    
# Đường dẫn đến thư mục test để load ảnh
image = Image.open('ca_neon_test.jpg')

#Cắt ảnh để đúng kích thước 64x64 pixel
size = (64, 64)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
image.show()
#Từ ảnh để tạo ra mảng
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load ảnh từ mảng trên
data[0] = normalized_image_array

#Chạy thuật toán và đưa ra kết quả
prediction = model.predict(data)
print(prediction)
m = prediction[0,0]
a= 0
array = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] # tổng cộng 15 classes
for i in array:
    if m < prediction[0,i]:
        m = prediction[0,i]
        a = i
print('vị trí của class giống nhất là:',a)
print('Giá trị :',m)