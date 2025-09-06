import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, \
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, \
    QStackedWidget, QRadioButton, QMessageBox, QDesktopWidget, QSlider, QScrollArea  # 添加 QScrollArea
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from LaClassify import predict_pic, getModelNum
from LaSegment import getMask, getImgs, recolor_and_display_image, getwrongMask
import numpy as np
import cv2
from PIL import Image, ImageQt
from LaLoadModel import mymodels, sam
import os


class ImageProcessingWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.file_name = ''
        self.tmp = 1

        self.init_ui()

    def init_ui(self):
        self.layout = QHBoxLayout(self)

        self.left_layout = QVBoxLayout()
        self.layout.addLayout(self.left_layout)

        self.right_layout = QVBoxLayout()
        self.layout.addLayout(self.right_layout)

        # Left Panel
        self.left_layout.addStretch()

        self.file_label = QLabel("请选择一张图片")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.file_label)

        self.left_layout.addStretch()

        self.file_path = QLabel("没有任何图片被选择")
        self.file_path.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.file_path)

        self.file_button = QPushButton("选择图片")
        self.file_button.clicked.connect(self.select_file)
        self.left_layout.addWidget(self.file_button)

        self.width_label = QLabel("长度:")
        self.height_label = QLabel("宽度:")
        self.unit_label = QLabel("单位:")

        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("输入长度(默认60)")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("输入宽度(默认40)")
        self.unit_combobox = QComboBox()
        self.unit_combobox.addItem("cm")
        self.unit_combobox.addItem("mm")
        self.unit_combobox.addItem("um")
        self.unit_combobox.addItem("nm")
        self.unit_combobox.setCurrentIndex(0)

        self.input_layout = QHBoxLayout()
        self.input_layout.addWidget(self.width_label)
        self.input_layout.addWidget(self.width_input)
        self.input_layout.addWidget(self.height_label)
        self.input_layout.addWidget(self.height_input)
        self.input_layout.addWidget(self.unit_label)
        self.input_layout.addWidget(self.unit_combobox)

        self.left_layout.addLayout(self.input_layout)

        # Right Panel
        self.classify_button = QPushButton("分类")
        self.classify_button.clicked.connect(self.show_classify_page)

        self.segment_button = QPushButton("分割")
        self.segment_button.clicked.connect(self.show_segment_page)

        self.guide_layout = QHBoxLayout()
        self.guide_layout.addWidget(self.classify_button)
        self.guide_layout.addWidget(self.segment_button)

        self.right_layout.addLayout(self.guide_layout)

        self.stacked_widget = QStackedWidget()
        self.right_layout.addWidget(self.stacked_widget)

        self.detail_page = QWidget()
        self.stacked_widget.addWidget(self.detail_page)

        self.classify_page = QWidget()
        self.stacked_widget.addWidget(self.classify_page)

        self.segment_page = QWidget()
        self.stacked_widget.addWidget(self.segment_page)

        self.stacked_widget.setCurrentWidget(self.detail_page)

        detail_layout = QVBoxLayout(self.detail_page)
        detail_label1 = QLabel('请在左侧打开一个岩石薄片或岩石标本图像')
        detail_label2 = QLabel('对于岩石薄片图像，点击上方"分类"按钮可以进行分类')
        detail_label3 = QLabel('对于岩石标本图像，点击上方"分割"按钮可以进行分割')
        detail_label4 = QLabel('左侧可以输入岩石标本图像的真实大小，便于分割后测量')
        detail_layout.addWidget(detail_label1)
        detail_layout.addWidget(detail_label2)
        detail_layout.addWidget(detail_label3)
        detail_layout.addWidget(detail_label4)

        classify_layout = QVBoxLayout(self.classify_page)

        classify_layout.addStretch()

        self.radio_layout = QHBoxLayout()
        self.radio_button1 = QRadioButton("MobileNet")
        self.radio_button2 = QRadioButton("Inception")
        self.radio_button3 = QRadioButton("DeiT")
        self.radio_button4 = QRadioButton("All")
        self.radio_layout.addWidget(self.radio_button1)
        self.radio_layout.addWidget(self.radio_button2)
        self.radio_layout.addWidget(self.radio_button3)
        self.radio_layout.addWidget(self.radio_button4)
        classify_layout.addLayout(self.radio_layout)
        classify_layout.addStretch()

        self.compute_button = QPushButton("开始预测")
        self.compute_button.clicked.connect(self.start_computation)
        classify_layout.addWidget(self.compute_button)
        classify_layout.addStretch()

        self.classify_label = QLabel("暂无预测结果")
        # 创建包含两个标签的水平布局
        classify_label1_layout = QHBoxLayout()
        classify_label2_layout = QHBoxLayout()
        classify_label3_layout = QHBoxLayout()
        classify_label4_layout = QHBoxLayout()
        classify_label5_layout = QHBoxLayout()
        # 第一个标签，保留原内容
        classify_label1_content = QLabel("板  岩：")
        classify_label2_content = QLabel("灰  岩：")
        classify_label3_content = QLabel("砂  岩：")
        classify_label4_content = QLabel("砾  岩：")
        classify_label5_content = QLabel("花岗岩：")
        # 第二个标签，添加"%"
        self.classify_label1_percent = QLabel("0.0%")
        self.classify_label2_percent = QLabel("0.0%")
        self.classify_label3_percent = QLabel("0.0%")
        self.classify_label4_percent = QLabel("0.0%")
        self.classify_label5_percent = QLabel("0.0%")
        # 将标签添加到布局中
        classify_label1_layout.addWidget(classify_label1_content)
        classify_label1_layout.addWidget(self.classify_label1_percent)
        classify_label2_layout.addWidget(classify_label2_content)
        classify_label2_layout.addWidget(self.classify_label2_percent)
        classify_label3_layout.addWidget(classify_label3_content)
        classify_label3_layout.addWidget(self.classify_label3_percent)
        classify_label4_layout.addWidget(classify_label4_content)
        classify_label4_layout.addWidget(self.classify_label4_percent)
        classify_label5_layout.addWidget(classify_label5_content)
        classify_label5_layout.addWidget(self.classify_label5_percent)
        # 将布局添加到分类布局中
        classify_layout.addLayout(classify_label1_layout)
        classify_layout.addLayout(classify_label2_layout)
        classify_layout.addLayout(classify_label3_layout)
        classify_layout.addLayout(classify_label4_layout)
        classify_layout.addLayout(classify_label5_layout)
        classify_layout.addWidget(self.classify_label)
        classify_layout.addStretch()

        # 分割页面布局
        segment_layout = QVBoxLayout(self.segment_page)

        slider_layout_iou = QHBoxLayout()  # IoU阈值布局
        slider_layout_nms = QHBoxLayout()  # 非最大抑制阈值布局

        # IoU阈值布局
        slider_label_iou = QLabel('预测IoU阈值：')
        slider_layout_iou.addWidget(slider_label_iou, 1)

        decrease_button_iou = QPushButton("-")
        decrease_button_iou.setFixedSize(20, 20)
        decrease_button_iou.clicked.connect(self.decrease_value_iou)
        slider_layout_iou.addWidget(decrease_button_iou, 1)

        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setMinimum(80)
        self.slider_iou.setMaximum(100)
        self.slider_iou.setTickInterval(1)
        self.slider_iou.setTickPosition(QSlider.TicksBelow)
        self.slider_iou.setSingleStep(1)
        self.slider_iou.setValue(100)
        self.slider_iou.valueChanged.connect(self.slider_value_changed_iou)
        slider_layout_iou.addWidget(self.slider_iou, 4)

        increase_button_iou = QPushButton("+")
        increase_button_iou.setFixedSize(20, 20)
        increase_button_iou.clicked.connect(self.increase_value_iou)
        slider_layout_iou.addWidget(increase_button_iou, 1)

        self.label_iou = QLabel(str(self.slider_iou.value() / 100))
        slider_layout_iou.addWidget(self.label_iou, 1)

        # 非最大抑制阈值布局
        slider_label_nms = QLabel('非最大抑制阈值：')
        slider_layout_nms.addWidget(slider_label_nms, 1)

        decrease_button_nms = QPushButton("-")
        decrease_button_nms.setFixedSize(20, 20)
        decrease_button_nms.clicked.connect(self.decrease_value_nms)
        slider_layout_nms.addWidget(decrease_button_nms, 1)

        self.slider_nms = QSlider(Qt.Horizontal)
        self.slider_nms.setMinimum(50)
        self.slider_nms.setMaximum(100)
        self.slider_nms.setTickInterval(5)
        self.slider_nms.setTickPosition(QSlider.TicksBelow)
        self.slider_nms.setSingleStep(5)
        self.slider_nms.setValue(70)
        self.slider_nms.valueChanged.connect(self.slider_value_changed_nms)
        slider_layout_nms.addWidget(self.slider_nms, 4)

        increase_button_nms = QPushButton("+")
        increase_button_nms.setFixedSize(20, 20)
        increase_button_nms.clicked.connect(self.increase_value_nms)
        slider_layout_nms.addWidget(increase_button_nms, 1)

        self.label_nms = QLabel(str(self.slider_nms.value() / 100))
        slider_layout_nms.addWidget(self.label_nms, 1)

        # 将布局添加到主布局
        segment_layout.addLayout(slider_layout_iou)
        segment_layout.addLayout(slider_layout_nms)

        self.segment_button_main = QPushButton('开始分割')
        self.segment_button_main.clicked.connect(self.start_segment)
        segment_layout.addWidget(self.segment_button_main)

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(300)  # 设置最小高度

        # 创建滚动区域的内容部件
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        # 将内容部件设置为滚动区域的部件
        self.scroll_area.setWidget(self.scroll_content)

        # 将滚动区域添加到主布局
        segment_layout.addWidget(self.scroll_area)

        # 初始化图像标签列表
        self.image_labels = []

    def clear_segment_layout(self):
        # 清除现有的图片和标签部件
        for image_label, kind_label, width_label, height_label in self.image_labels:
            # 从布局中移除部件
            self.scroll_layout.removeWidget(image_label)
            self.scroll_layout.removeWidget(kind_label)
            self.scroll_layout.removeWidget(width_label)
            self.scroll_layout.removeWidget(height_label)
        
            # 删除部件
            image_label.deleteLater()
            kind_label.deleteLater()
            width_label.deleteLater()
            height_label.deleteLater()
    
        # 清空列表
        self.image_labels = []

    def start_segment(self):
        if self.segment_button_main.text() == '还原':
            self.select_file()
            self.segment_button_main.setText('开始分割')
            return
        
        # 清除现有的分割结果
        self.clear_segment_layout()

        real_width = self.width_input.text()
        real_height = self.height_input.text()
        if real_width:
            if not real_width.isdigit():
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '长的值输入错误')
                msg_box.exec_()
                return
            else:
                real_width = int(real_width)
        else:
            real_width = 60
        if real_height:
            if not real_height.isdigit():
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '宽的值输入错误')
                msg_box.exec_()
                return
            else:
                real_height = int(real_height)
        else:
            real_height = 40

        print(real_width, real_height)

        self.segment_button_main.setText("分割中...")
        self.segment_button_main.setEnabled(False)
        QApplication.processEvents()

        iou_threshold = self.slider_iou.value() / 100
        nms_threshold = self.slider_nms.value() / 100

        file_image = cv2.imread(self.file_name)
        file_image = cv2.cvtColor(file_image, cv2.COLOR_BGR2RGB)
        masks = getMask(sam, file_image, iou_threshold, nms_threshold)
        #masks = getwrongMask(file_image, self.tmp)
        #self.tmp += 1
        imgs = getImgs(file_image, masks)
        img = recolor_and_display_image(file_image, masks)

        cv2.imwrite('{}tmp.jpg'.format(self.file_name.split('.jpg')[0]), img)
        self.file_path.setText('{}tmp.jpg'.format(self.file_name.split('.jpg')[0]))
        pixmap = QPixmap('{}tmp.jpg'.format(self.file_name.split('.jpg')[0]))
        self.file_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

        self.segment_button_main.setText("预测中...")
        self.segment_button_main.setEnabled(False)
        QApplication.processEvents()

        kinds = []
        for img in imgs:
            image = Image.fromarray(img)
            model_num = getModelNum(4)
            models = [mymodels[num] for num in model_num]
            predictions = predict_pic(image, models)
            print(predictions)
            kinds.append(np.argmax(predictions))

        print("修改布局")

        # 生成并添加新的图片和标签部件
        double_layout = QHBoxLayout()
        k = 0
        for img, kind in zip(imgs, kinds):
            stones = ['背景','粉砂岩','花岗岩','灰岩','粒岩','泥岩']
            if kind != 0:
                single_layout = QVBoxLayout()
        
                image_label = QLabel()
        
                image_cropped = self.crop_white_border(img)
                cropped_width, cropped_height = image_cropped.size
                height, width, channels = img.shape
                width_ratio = cropped_width / width
                height_ratio = cropped_height / height
        
                # 使用正确的方法转换图像
                try:
                    qimage = ImageQt.ImageQt(image_cropped)
                    pixmap = QPixmap.fromImage(qimage)
                except:
                    img_array = np.array(image_cropped)
                    height, width, channel = img_array.shape
                    bytes_per_line = 3 * width
                    qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage)
        
                image_label.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio))
                single_layout.addWidget(image_label)
        
                kind_label = QLabel(f"类别： {stones[kind]}")
                single_layout.addWidget(kind_label)
        
                unit = self.unit_combobox.currentText()
                width_label = QLabel(f"长度： {int(real_width*width_ratio)} {unit}")
                single_layout.addWidget(width_label)
        
                height_label = QLabel(f"宽度： {int(real_height*height_ratio)} {unit}")
                single_layout.addWidget(height_label)
        
                double_layout.addLayout(single_layout)
                if k % 2 == 1:
                    self.scroll_layout.addLayout(double_layout)
                    double_layout = QHBoxLayout()
                k += 1
        
                self.image_labels.append((image_label, kind_label, width_label, height_label))

        # 添加最后一个双布局（如果包含内容）
        if double_layout.count() > 0:
            self.scroll_layout.addLayout(double_layout)
    
        # 添加一个拉伸项，确保内容从顶部开始
        self.scroll_layout.addStretch()
        
        self.segment_button_main.setText("开始分割")
        self.segment_button_main.setEnabled(True)

    def start_computation(self):
        image = Image.open(self.file_name)

        if self.radio_button1.isChecked():
            self.compute_button.setText("计算中...")
            self.compute_button.setEnabled(False)
            QApplication.processEvents()
            model_num = getModelNum(0)
            models = [mymodels[num] for num in model_num]
            predictions = predict_pic(image, models)
        elif self.radio_button2.isChecked():
            self.compute_button.setText("计算中...")
            self.compute_button.setEnabled(False)
            QApplication.processEvents()
            model_num = getModelNum(1)
            models = [mymodels[num] for num in model_num]
            predictions = predict_pic(image, models)
        elif self.radio_button3.isChecked():
            self.compute_button.setText("计算中...")
            self.compute_button.setEnabled(False)
            QApplication.processEvents()
            model_num = getModelNum(2)
            models = [mymodels[num] for num in model_num]
            predictions = predict_pic(image, models)
        elif self.radio_button4.isChecked():
            self.compute_button.setText("计算中...")
            self.compute_button.setEnabled(False)
            QApplication.processEvents()
            model_num = getModelNum(3)
            models = [mymodels[num] for num in model_num]
            predictions = predict_pic(image, models)
        else:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '未选择任何模型')
            msg_box.exec_()
            return

        print(predictions)
        self.classify_label1_percent.setText(str(round(predictions[0] * 100, 2)) + '%')
        self.classify_label2_percent.setText(str(round(predictions[1] * 100, 2)) + '%')
        self.classify_label3_percent.setText(str(round(predictions[2] * 100, 2)) + '%')
        self.classify_label4_percent.setText(str(round(predictions[3] * 100, 2)) + '%')
        self.classify_label5_percent.setText(str(round(predictions[4] * 100, 2)) + '%')
        print('over')
        stones = ['板  岩', '灰  岩', '砂  岩', '砾  岩', '花岗岩']
        self.classify_label.setText('预测结果为 : '+stones[np.argmax(predictions)])

        self.compute_button.setText("开始预测")
        self.compute_button.setEnabled(True)

    def show_classify_page(self):
        if self.file_name == '':
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请先选择文件')
            msg_box.exec_()
            return
        self.stacked_widget.setCurrentWidget(self.classify_page)

    def show_segment_page(self):
        if self.file_name == '':
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请先选择文件')
            msg_box.exec_()
            return
        self.stacked_widget.setCurrentWidget(self.segment_page)

    def act_select_file(self):
        if self.file_name:
            self.file_path.setText(self.file_name)
            pixmap = QPixmap(self.file_name)
            self.file_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        self.stacked_widget.setCurrentWidget(self.segment_page)

        # 清除现有的图片和标签部件
        for image_label, kind_label, width_label, height_label in self.image_labels:
            self.scroll_layout.removeWidget(image_label)
            self.scroll_layout.removeWidget(kind_label)
            self.scroll_layout.removeWidget(width_label)
            self.scroll_layout.removeWidget(height_label)
            image_label.deleteLater()
            kind_label.deleteLater()
            width_label.deleteLater()
            height_label.deleteLater()
        self.image_labels = []

        self.classify_label1_percent.setText("0.0%")
        self.classify_label2_percent.setText("0.0%")
        self.classify_label3_percent.setText("0.0%")
        self.classify_label4_percent.setText("0.0%")
        self.classify_label5_percent.setText("0.0%")
        self.classify_label.setText("暂无预测结果")

    def select_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "./images", "JPEG Files (*.jpg)", options=options)
        if self.file_name:
            self.file_path.setText(self.file_name)
            pixmap = QPixmap(self.file_name)
            self.file_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        self.stacked_widget.setCurrentWidget(self.detail_page)

        # 清除现有的图片和标签部件
        for image_label, kind_label, width_label, height_label in self.image_labels:
            self.scroll_layout.removeWidget(image_label)
            self.scroll_layout.removeWidget(kind_label)
            self.scroll_layout.removeWidget(width_label)
            self.scroll_layout.removeWidget(height_label)
            image_label.deleteLater()
            kind_label.deleteLater()
            width_label.deleteLater()
            height_label.deleteLater()
        self.image_labels = []

        self.classify_label1_percent.setText("0.0%")
        self.classify_label2_percent.setText("0.0%")
        self.classify_label3_percent.setText("0.0%")
        self.classify_label4_percent.setText("0.0%")
        self.classify_label5_percent.setText("0.0%")
        self.classify_label.setText("暂无预测结果")

    def slider_value_changed_iou(self):
        self.label_iou.setText(str(self.slider_iou.value() / 100))

    def decrease_value_iou(self):
        if self.slider_iou.value() > self.slider_iou.minimum():
            self.slider_iou.setValue(self.slider_iou.value() - 1)

    def increase_value_iou(self):
        if self.slider_iou.value() < self.slider_iou.maximum():
            self.slider_iou.setValue(self.slider_iou.value() + 1)

    def slider_value_changed_nms(self):
        self.label_nms.setText(str(self.slider_nms.value() / 100))

    def decrease_value_nms(self):
        if self.slider_nms.value() > self.slider_nms.minimum():
            self.slider_nms.setValue(self.slider_nms.value() - 1)

    def increase_value_nms(self):
        if self.slider_nms.value() < self.slider_nms.maximum():
            self.slider_nms.setValue(self.slider_nms.value() + 1)

    def crop_white_border(self, image):
        image = Image.fromarray(image)
        width, height = image.size
        left = width
        right = 0
        top = height
        bottom = 0
        # 遍历边缘像素
        for x in range(width):
            for y in range(height):
                r, g, b = image.getpixel((x, y))
                # 检查是否为白色
                if r != 255 or g != 255 or b != 255:
                    # 更新边界框坐标
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
        # 裁剪图像
        image_cropped = image.crop((left, top, right, bottom))
        return image_cropped


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    image_processing_widget = ImageProcessingWidget()
    window.setCentralWidget(image_processing_widget)
    window.setWindowTitle("基于深度学习技术的岩石分类识别系统")
    screen = QDesktopWidget().screenGeometry()
    size = window.geometry()
    window.setGeometry(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2), 800, 450)
    window.show()

    sys.exit(app.exec_())
