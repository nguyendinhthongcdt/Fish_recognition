# Thêm các thư viện cần thiết
from tkinter import*
import cv2
from PIL import Image, ImageTk
import time
import numpy as np
import tensorflow as tf
from keras.models import load_model

physical_devices = tf.config.list_physical_devices("GPU")
threshold = 0.75

# Tải model
model = load_model('C:\\Users\\dinht\\Desktop\\Fish_recognize\\model.h5')

class TestVideo:
    def __init__(self, video_source = 0):
        self.appName = "Fish Recognition"  
        self.window = Tk()  
        self.window.title(self.appName)
        self.window.geometry("1080x720")
        self.window.configure(bg = 'white')

        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.Top_frame = Frame(self.window, bg = "white", width = 1080, height = 720)
        self.Top_frame.place(x = 0, y= 0)

        #Chèn backgound, logo
        
        image_1 = Image.open("C:/Users/dinht/Desktop/AI_Project/bg2.jpg")
        resize_image = image_1.resize((1080,720), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(resize_image)       
        image_3 = Image.open("C:/Users/dinht/Desktop/AI_Project/hcmute3.png")
        resize_image2 = image_3.resize((120,120), Image.ANTIALIAS)
        img3 = ImageTk.PhotoImage(resize_image2)
        image_4 = Image.open("C:/Users/dinht/Desktop/AI_Project/fme2.jpg")
        resize_image4 = image_4.resize((120,120), Image.ANTIALIAS)
        img4 = ImageTk.PhotoImage(resize_image4)
        
        self.label_1 = Label(self.Top_frame, image = img1)
        self.label_1.place(x = 0, y = 120)
        self.label_2 = Label(self.Top_frame, text = "NHẬN DẠNG CÁC LOẠI CÁ CẢNH", font = "arial 28 bold", bg = 'yellow', fg = 'blue')
        self.label_2.place(x = 360, y = 50)        
        self.label_3 = Label(self.Top_frame, image = img3)
        self.label_3.place(x = 0, y = 0)
        self.label_4 = Label(self.Top_frame, image = img4)
        self.label_4.place(x = 120, y = 0)


       # Tạo nút nhấn
        self.btn_capture = Button(self.window, text = "DỰ ĐOÁN", font = 30, command = self.predict)
        self.btn_capture.place(x = 830 , y = 480)

        # Tạo text
        self.text_1 = Text(self.window, bg = "white", font ="arial 18 bold")
        self.text_1.place(x = 670, y= 300, width = 400, height = 40)
        self.text_2 = Text(self.window, bg = "white", font ="arial 18 bold")
        self.text_2.place(x = 670, y= 400, width = 400, height = 40)

        # Create a canvas
        self.canvas_1 = Canvas(self.window, width = self.vid.width,height = self.vid.height, bg= 'white')
        self.canvas_1.place(x = 20, y = 200)

        self.update()
        self.window.mainloop()
        self.predict()

    def update(self):
        # Lấy khung ảnh
        isTrue, frame = self.vid.getFrame()
        if isTrue:             
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas_1.create_image(0, 0, image = self.photo, anchor = NW)
        self.window.after(15, self.update)

    def predict(self):
        isTrue, frame = self.vid.getFrame()
        if isTrue:             
            img = np.asarray(frame)          
            img = cv2.resize(img,(64,64))
            img = img/255
            img = img.reshape(1,64,64,3)
        
            predictions = model.predict(img)
            classIndex = np.argmax(predictions, axis=-1)
            probabilityValue = np.amax(predictions)                  
            
            if probabilityValue > threshold:
                print(self.getClassName(classIndex))
                self.text_1.delete('1.0', END)
                self.text_1.insert('1.0', self.getClassName(classIndex)+"\n")
                self.text_2.delete('1.0', END)
                self.text_2.insert('1.0',"Độ chính xác : " + str(round(probabilityValue*100, 2)) + " %"+"\n")              
            
        
    def getClassName(self,classNo):
        if classNo == 0:
            return 'Kết quả: Cá ba đuôi'
        elif classNo == 1:
            return 'Kết quả: Cá Cánh Buồm hồng'
        elif classNo == 2:
            return 'Kết quả: Cá chép Sư Tử Trắng'
        elif classNo == 3:
            return 'Kết quả: Cá Hạc Đỉnh Hồng'
        elif classNo == 4:
            return 'Kết quả: Cá Hổ Bạc'
        elif classNo == 5:
            return 'Kết quả: Cá La Hán'
        elif classNo == 6:
            return 'Kết quả: Cá Mã Giáp Hoàng Kim'
        elif classNo == 7:
            return 'Kết quả: Cá Mún Panda'
        elif classNo == 8:
            return 'Kết quả: Cá Neon'
        elif classNo == 9:
            return 'Kết quả: Cá Phượng Hoàng'
        elif classNo == 10:
            return 'Kết quả: Cá rồng Huyết Long'
        elif classNo == 11:
            return 'Kết quả: Cá rồng Kim Long'
        elif classNo == 12:
            return 'Kết quả: Cá sam Black Diamond'
        elif classNo == 13:
            return 'Kết quả: Cá Thần tiên'
        elif classNo == 14:
            return 'Kết quả: Cá Tứ Vân'

     
class MyVideoCapture:
    def __init__(self, video_source):
        # Mở video
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Không thể mở Camera\n", video_source)

        # Set kích thước khung video
        self.vid.set(3, 450)
        self.vid.set(4, 450)
        self.width = self.vid.get(3)
        self.height = self.vid.get(4)

    def getFrame(self):
        if self.vid.isOpened():
            isTrue, frame = self.vid.read()
            if isTrue:
    
                return (isTrue, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (isTrue, None)
            
        else:
            return (isTrue, None)

        
    def __def__(self):
        if self.vid.isOpened():
            self.vid.release()
        
if __name__ == "__main__":
    TestVideo()