# Thêm các thư viện cần thiết
import cv2
import time

# Tách ảnh từ video
def main():
    cap = cv2.VideoCapture('C:\\Users\\dinht\\Desktop\\Fish_recognize\\video\\ca_neon.mp4')
    time.sleep(1)
    if cap is None or not cap.isOpened():
        print('Khong the mo file video')
        return
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE);
    n = 1
    dem = 200
    while True:
        [success, imgROI] = cap.read()
        ch = cv2.waitKey(30)
        if success:
            imgROI=cv2.resize(imgROI,(150,150))
            cv2.imshow('Image', imgROI)
        else:
            break
        if n%4 == 0:
            
            # Lưu ảnh vào thư mục data train
            filename = 'C:\\Users\\dinht\\Desktop\\Fish_recognize\\data_train\\ca_neon\\ca_neon_train_%04d.jpg'%(dem)
            cv2.imwrite(filename,imgROI)
            dem = dem + 1
        n = n + 1
    return

if __name__ == "__main__":
    main()
