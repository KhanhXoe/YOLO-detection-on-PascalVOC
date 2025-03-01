# Triển khai YOLOv1 cho bài toán Detection 20 đối tượng
## Tổng quan
Kho lưu trữ này chứa mã nguồn Python triển khai mô hình phát hiện đối tượng YOLOv1 (You Only Look Once) từ đầu, sử dụng PyTorch. Mã bao gồm các thành phần như bộ tải dữ liệu, kiến trúc mô hình, hàm mất mát và các phép đo đánh giá như Intersection over Union (IoU), Non-Maximum Suppression (NMS), và Mean Average Precision (mAP). Triển khai này được thiết kế để hoạt động với tập dữ liệu PASCAL VOC, một tập dữ liệu phổ biến cho các bài toán phát hiện đối tượng.
## Tính năng 
- __Bộ tải dữ liệu:__ Lớp VOCDataset để tải hình ảnh và nhãn từ tập dữ liệu __PASCAL VOC__.
- __Mô hình YOLOv1:__ Kiến trúc YOLOv1 được triển khai đầy đủ với các tầng tích chập và tầng kết nối đầy đủ.
- __Hàm mất mát:__ Lớp YOLO_Loss tùy chỉnh tính toán hàm mất mát nhiều thành phần theo bài báo YOLO gốc.
- __Phép đo đánh giá:__ Bao gồm IoU, NMS và mAP để đánh giá hiệu suất mô hình.
- __Tiện ích:__ Các hàm để trực quan hóa hộp giới hạn, chuyển đổi dự đoán và lưu/tải checkpoint của mô hình.

## Yêu cầu về thư viện
- Python 3.7 trở lên
- PyTorch
- numpy
- pandas
- matplotlib
- OpenCV (cv2)
- PIL (Pillow)
- tqdm

## Cấu trúc file 
    ├── README.md              # Tài liệu dự án
    ├── requirements.txt       # Danh sách các thư viện phụ thuộc
    ├── yolov1.ipynb           # Notebook Jupyter chính chứa mã triển khai
    ├── data/                  # Thư mục chứa tập dữ liệu PASCAL VOC (không bao gồm trong repo)
    │   ├── images/            # Thư mục chứa các file ảnh
    │   ├── labels/            # Thư mục chứa các file nhãn
    │   └── annotations.csv    # File CSV ánh xạ tên ảnh và nhãn
    ├── checkpoints/           # Thư mục lưu checkpoint của mô hình
       └── checkpoint_model.tar  # File checkpoint mẫu

## Hướng dẫn sử dụng
1. __Chuẩn bị dữ liệu__
    - Tải tập dữ liệu PASCAL VOC (ví dụ: VOC2007 hoặc VOC2012).
    - Sắp xếp dữ liệu vào data/images/ (cho ảnh) và data/labels/ (cho nhãn).
    - Tạo file annotations.csv với hai cột: image (tên file ảnh) và label (tên file nhãn tương ứng).
2. __Chạy mã nguồn__
    - Mở file yolov1.ipynb trong Jupyter Notebook hoặc JupyterLab.
    - Thực thi lần lượt các ô mã để tải dữ liệu, định nghĩa mô hình và triển khai quy trình huấn luyện/đánh giá.
3. __Train__
    - Điều chỉnh các siêu tham số (ví dụ: S, B, C, tốc độ học) nếu cần.
    - Sử dụng lớp YOLO_Loss và một bộ tối ưu (ví dụ: torch.optim.Adam) để huấn luyện mô hình.
4. __Đánh giá model__
    - Sử dụng hàm get_bboxes để tạo dự đoán và hộp giới hạn thực tế.
    - Tính mAP bằng hàm mean_average_precision.
5. __Trực quan kết quả__
    - Sử dụng hàm plot_image để hiển thị các hộp giới hạn dự đoán trên ảnh kiểm tra.

## Kiến trúc model
YOLOv1 bao gồm:

1. Loạt tích chập (Darknet) Backbone định nghĩa trong biến architecture_config.
2. Các tầng kết nối đầy đủ để xuất dự đoán theo định dạng S x S x (B * 5 + C), trong đó:
    - S: Kích thước lưới (mặc định: 7).
    - B: Số hộp giới hạn trên mỗi ô lưới (mặc định: 2).
    - C: Số lớp (mặc định: 20 cho PASCAL VOC).
3. Hàm mất mát:
- Sai số tọa độ (x, y, chiều rộng, chiều cao).
- Độ tin cậy của đối tượng khi có đối tượng.
- Độ tin cậy "không có đối tượng" khi không có đối tượng.
- Sai số phân loại.

        # Định nghĩa dataset và dataloader
        dataset = VOCDataset(
        csv_file="data/annotations.csv", img_dir="data/images/",
        label_dir="data/labels/", S=7, B=2, C=20, transforms=None
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Khởi tạo mô hình
        model = YOLOv1(S=7, B=2, C=20).to(device)

        # Định nghĩa hàm mất mát và bộ tối ưu
        criterion = YOLO_Loss(S=7, B=2, C=20)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Vòng lặp huấn luyện (đơn giản hóa)
        for epoch in range(num_epochs):
            for batch_idx, (images, targets) in enumerate(loader):
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

# Hạn chế
- Triển khai này giả định ảnh đầu vào được điều chỉnh về kích thước cố định (ví dụ: 448x448) theo bài báo YOLOv1 gốc.
- Việc huấn luyện từ đầu có thể yêu cầu tài nguyên tính toán lớn và thời gian dài.
- Mã không bao gồm trọng số được huấn luyện trước; bạn cần huấn luyện mô hình hoặc điều chỉnh để sử dụng backbone đã được huấn luyện trước (ví dụ: từ torchvision).
