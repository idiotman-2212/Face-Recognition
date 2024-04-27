# Face-Recognition
### Nhận diện khuôn mặt bằng MTCNN và Facenet

# Tìm hiểu khái niệm
- MTCNN là viết tắt của Multi-task Cascaded Convolutional Networks. Nó là bao gồm 3 mạng CNN xếp chồng và đồng thời hoạt động khi detect khuôn mặt. Mỗi mạng có cấu trúc khác nhau và đảm nhiệm vai trò khác nhau trong task. Đầu ra của MTCNN là vị trí khuôn mặt và các điểm trên mặt như: mắt, mũi, miệng…
- Facenet là của ông Google giới thiệu năm 2015, và thằng model này thì mình cứ nhét ảnh vào (đúng size của nó) thì nó trả ra 1 vector 128 features cho 1 khuôn mặt. Sau đó dùng SVM để phân nhóm các vector đó vào các nhóm để biết vector đó là mặt của ai.
![image](https://github.com/idiotman-2212/Face-Recognition/assets/82036270/4c2674c8-bc14-493f-b2c2-b3502447c202)

# Cài đặt chương trình 
- Tạo thư mục MiAI_FaceRecog_3 và clone từ github về máy tính.
- Tạo thư mục Dataset trong MiAI_FaceRecog_3, trong đó tạo tiếp thư mục FaceData và dưới FaceData là tạo tiếp 2 thư mục raw và processed.
- Tạo thư mục Models trong MiAI_FaceRecog_3 để chờ sẵn tẹo lưu model sau.
![image](https://github.com/idiotman-2212/Face-Recognition/assets/82036270/cddc46c3-b909-4025-b8e9-263f4ad0ee96)

# Chuẩn bị ảnh khuôn mặt để train
- Sưu tầm ảnh của 2 người trở lên, mỗi người 10 tấm hình rõ mặt (tạm chấp nhận yêu cầu hiện tại của bài này là at least 2 người nhé, mình sẽ tìm hiểu thêm sau). Mình ví dụ 5 người tên là Anh, dien, Nhut Truong, Truong và Tuan Anh nhé. Các bạn tạo 5 thư mục Anh, dien, Nhut Truong, Truong và Tuan Anh trong thư mục raw và copy ảnh của 5 người vào riêng 5 thư mục đó, ảnh của ai vào thư mục của người đó nhé. Hoặc bạn có thể tạo 100 ảnh dựa vào webcam bằng cách chạy lệnh:<br>
<pre>python capture.py</pre>
![image](https://github.com/idiotman-2212/Face-Recognition/assets/82036270/a15a997e-8772-4f11-9f26-7582022b573f)

![image](https://github.com/idiotman-2212/Face-Recognition/assets/82036270/7bbb404c-357d-484f-b392-0278a6c6cfb4)

Chú ý: Trong các ảnh sưu tầm, chỉ có đúng 1 khuôn mặt của người đó, không được có quá 1 khuôn mặt/ảnh.
Ví dụ cây thư mục của mình để các bạn tham khảo:<br>
<pre>
  -FaceData
   |-processed
   |   |-Anh
   |   |-dien
   |   |-Nhut Truong
   |   |-Truong
   |   |-Tuan Anh
   |-raw
       |-Anh
       |-dien
       |-Nhut Truong
       |-Truong
       |-Tuan Anh
</pre>

# Cài đặt các thư viện cần thiết
Các bạn đứng ở thư mục gốc là MiAI_FaceRecog_3 chạy lệnh sau để cài tất cả các thư viện cần thiết: 
<br>
<pre>pip install -r requirements.txt </pre>

# Tiền xử lý dữ liệu để cắt khuôn mặt từ ảnh gốc
- Với chỗ ảnh đã sưu tầm bên trên, có thể là ảnh cả người, bây giờ chúng ta sẽ cắt riêng khuôn mặt ra để train nhé. Chuyển về thư mục MiAI_FaceRecog_3 và chạy lệnh:<br>
<pre> python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25</pre>
- Chạy xong thấy nó hiển thị dạng “Total number of images: …” là thành công rồi. Các bạn để ý sẽ thấy có thêm thư mục processed có cấu trúc tương tự thư mục raw nhưng chỉ chứa dữ liệu khuôn mặt đã được xử lý. Ví dụ như ảnh dưới:
![image](https://github.com/idiotman-2212/Face-Recognition/assets/82036270/d69e86d4-d32f-4dae-99e2-25e56dd9cf73)

# Tải dữ liệu pretrain của Facenet về máy
- Các bạn tải weights pretrain về tại link này: Tại **đây** (https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view). Sau khi tải xong về, các bạn copy toàn bộ file tải về vào thư mục Models, chú ý chỉ lấy file, bỏ hết các thư mục như hình bên dưới của mình.
![image](https://github.com/idiotman-2212/Face-Recognition/assets/82036270/b2dabad2-c4b7-4913-bd6f-897097de0494)
# Train model để nhận diện khuôn mặt.
- Chuyển về thư mục MiAI_FaceRecog_3 nếu đang đứng ở thư mục khác nhé. Sau đó chạy lệnh train: <br>
<pre>python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000</pre> <br>
- Khi nào màn hình hiện lên chữ “Saved classifier model to file “Models/facemodel.pkl” là xong.
# Chạy chương trình.
- Các bạn chạy file face_rec_cam.py bằng lệnh sau: <br>
<pre> python src/face_rec_cam.py</pre>
- Kết quả:
![image](https://github.com/idiotman-2212/Face-Recognition/assets/82036270/004a53ed-75b1-48e4-90f9-b45c08632d71)
- Nhận diện qua video. Chạy lệnh: <br>
<pre> python src/face_rec.py --path video/camtest.mp4</pre>

# Tài liệu tham khảo
https://miai.vn/2019/09/11/face-recog-2-0-nhan-dien-khuon-mat-trong-video-bang-mtcnn-va-facenet/
