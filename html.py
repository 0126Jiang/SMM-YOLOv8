import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO

# 加载 YOLOv8n 模型
model = YOLO('yolov8n.pt')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("口腔科曲面断层片检测")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建界面组件
        self.initUI()
    
    def initUI(self):
        # 主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 上传按钮
        self.upload_button = QPushButton("上传曲面断层片")
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)
        
        # 图像显示标签
        self.image_label = QLabel("请上传图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        
        # 检测结果显示标签
        self.result_label = QLabel("检测结果将显示在这里")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)
    
    def upload_image(self):
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg)")
        if file_path:
            # 显示上传的图像
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))
            
            # 进行推理
            results = self.predict_image(file_path)
            
            # 显示检测结果
            self.display_results(results)
    
    def predict_image(self, image_path):
        # 使用 YOLOv8n 模型进行推理
        results = model(image_path)
        
        # 解析结果
        detections = results[0].boxes.data.cpu().numpy()
        labels = results[0].names
        confidences = results[0].boxes.conf.cpu().numpy()
        
        result_text = []
        for i, box in enumerate(detections):
            x1, y1, x2, y2, conf, cls = box
            label = labels[int(cls)]
            result_text.append(f"检测到 {label}，置信度：{conf:.2f}，位置：({x1}, {y1}) - ({x2}, {y2})")
        
        return result_text
    
    def display_results(self, results):
        if results:
            result_text = "
".join(results)
            self.result_label.setText(result_text)
        else:
            self.result_label.setText("未检测到任何异常")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())