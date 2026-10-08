# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Tô Huy Thông |
| MSSV | 2A202602608 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/thong0609/K4-L3-DAY21-ToHuyThong-2A202602608-CI-CD-for-AI-Systems |
| Ngày nộp | 08/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.710900 | 0.878000 |
| 2 | 50 | 0.05 | 2 | 0.605128 | 0.846000 |
| 3 | 200 | 0.1 | 5 | 0.714932 | 0.874000 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần 3 đạt F1 cao nhất (0.714932), vượt ngưỡng 0.65. Các lần chạy dùng cùng dữ liệu và random_state=42. Lần 1 có accuracy cao nhất (0.878) nhưng F1 thấp hơn, nên chọn theo F1. Lần 2 dùng ít cây, learning_rate nhỏ và cây nông cho kết quả thấp nhất. Giảm learning_rate thường cần tăng số cây; các thí nghiệm thay đổi đồng thời nhiều tham số nên chưa tách được tác động riêng. Cấu hình tốt nhất đã lưu vào params.yaml.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập Adult có khoảng 24,8% mẫu thu nhập trên 50K. Vì vậy, mô hình luôn dự đoán thu nhập thấp vẫn đạt accuracy khoảng 75,2%, dù không nhận ra mẫu dương nào. F1 lớp dương kết hợp precision và recall, phản ánh khả năng phát hiện người thu nhập cao và hạn chế dự đoán dương sai. Lab dùng `f1_score(y_eval, preds)` với lớp dương mặc định là 1. Không dùng weighted F1 vì lớp đông chi phối điểm trung bình; macro F1 cân bằng trọng số hai lớp nhưng vẫn không đo riêng lớp dương. Pipeline chỉ triển khai khi F1 đạt ít nhất 0.65. Kết quả thực tế cũng cho thấy accuracy cao nhất không đồng nghĩa F1 cao nhất.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Mã ban đầu chưa dùng được với AWS. | Khung bài sử dụng SDK và DVC remote của GCP. | Tôi đổi sang boto3, dvc[s3] và S3 ở Sydney. |
| Thư viện lưu trữ bị xung đột khi chuyển cloud. | gcsfs cũ yêu cầu phiên bản fsspec khác với thư viện S3. | Tôi gỡ dvc-gs, gcsfs và kiểm tra lại bằng pip check. |
| Mẫu pipeline ghi đè model trước khi kiểm tra F1. | Bước upload model nằm trong job Train. | Tôi chuyển upload sang Release sau Quality Gate; cả bốn jobs đã xanh. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (22.361 mẫu) | 0.714932 | 0.874000 |
| Bước 3 (44.722 mẫu) | **0.735426** | **0.882000** |

**Nhận xét:** Sau khi bổ sung 22.361 mẫu từ `train_batch2`, F1 tăng từ 0.714932 lên 0.735426 (+0.0205) và accuracy từ 0.874 lên 0.882. Điều này cho thấy dữ liệu mới cùng phân phối với batch đầu (cùng nguồn Adult Dataset) nên mô hình học thêm được thông tin hữu ích, dẫn đến cải thiện nhẹ. Trong trường hợp hai batch cùng phân phối, gấp đôi dữ liệu thường chỉ dao động trong khoảng nhỏ — kết quả ở đây là tích cực, F1 tăng và vẫn vượt ngưỡng 0.65. Quan trọng hơn cả, pipeline tự động chạy lại hoàn toàn khi commit dữ liệu mới, không cần thao tác thủ công nào.
