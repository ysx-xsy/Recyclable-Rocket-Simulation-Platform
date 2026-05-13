import sys
import image_rc
import numpy as np
import pandas as pd
import os
import json
import cv2
import matplotlib.pyplot as plt
from inference_pid import PIDInference
from inference_pid_RL import PIDRLInference
from PySide6.QtWidgets import QApplication, QWidget, QFileDialog, QMessageBox, QSplashScreen
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QIcon
from montecarlo_pid import monte_carlo_pid
from my_spinbox import FreeInputSpinBox

class RocketApp(QWidget):
    def __init__(self):
        super().__init__()
        
        # 1. 加载 UI 文件
        loader = QUiLoader()
        loader.registerCustomWidget(FreeInputSpinBox)
        ui_file = QFile("rocket.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"无法打开 UI 文件: {ui_file.errorString()}")
            return
        self.ui = loader.load(ui_file, self)
        self.setWindowIcon(QIcon(":/icons/icons/rocket.png"))
        ui_file.close()

        # 将 UI 中的主要控件映射为类属性方便调用
        self.stacked_widget = self.ui.stackedWidget
        self.lists = [
            (self.ui.listWidget, 0),    # 起始索引 0 (Homepage, Option, Simulation)
            (self.ui.listWidget_2, 3),  # 起始索引 3 (Collection, Evaluation, Comparison)
            (self.ui.listWidget_3, 6)   # 起始索引 6 (Help, Advice)
        ]

        # 2. 绑定信号与槽
        self.setup_navigation()

        # 3. 程序启动默认显示 Homepage
        self.go_to_homepage()

        # 1. 初始化你的物理模拟逻辑
        #self.inference_RL = RocketInference()
        self.inference_pid = PIDInference()
        self.inference_pid_RL = PIDRLInference()
        self.init_set = {'off_centering': self.ui.doubleSpinBox.value(), 'initial_height': self.ui.doubleSpinBox_2.value(), 'horizontal_velocity': self.ui.doubleSpinBox_19.value(), 'vertical_velocity': self.ui.doubleSpinBox_18.value(), 'pitch_angle': np.deg2rad(self.ui.doubleSpinBox_3.value()), 'wind_strength': self.ui.doubleSpinBox_6.value()}
        self.state_buffer = []
        self.seed = None

        # 2. 设置定时器 (例如 30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_one_step)
        self.ui.pushButton.clicked.connect(self.start_simulation)
        self.ui.pushButton_4.clicked.connect(self.pause_simulation)
        self.ui.pushButton_3.clicked.connect(self.continue_simulation)
        self.ui.pushButton_2.clicked.connect(self.abort_simulation)

        self.max_steps = 2000
        self.step_count = 0
        self.show_flame = False # 用于切换 frame_0/1

        self.ui.comboBox.setStyleSheet("""
        /* 1. 下拉框整体样式 */
        QComboBox {
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 1px 18px 1px 3px;
        }

        /* 2. 关键：下拉列表视图（解决重影和背景色） */
        QComboBox QAbstractItemView {
            background-color: white;   /* 强制背景为白色 */
            border: 1px solid #999;    /* 给下拉框加一个清晰的边框 */
            selection-background-color: #0078d7; /* 选中时的蓝色背景 */
            selection-color: white;    /* 选中时的文字颜色 */
            outline: none;             /* 去掉虚线框 */
        }
        """)

        self.analysis_pid = monte_carlo_pid()
        self.number = 0
        self.ifRL = False
        self.ifPID = False
        self.ifPIDRL = False
        self.two_d = False
        self.one_d = False
        self.four_d = False
        self.choice = None
        self.choice_2 = None
        self.seed = None
        text = self.ui.comboBox.currentText()
        self.handle_selection(text)
        self.simulation_number()
        self.off_centering_range_val = None
        self.initial_height_range_val = None
        self.horizontal_velocity_range_val = None
        self.vertical_velocity_range_val = None
        self.pitch_angle_range_val = None
        self.wind_strength_range_val = None
        self.initial_height_range()
        self.off_centering_range()
        self.wind_strength_range()
        self.pitch_angle_range()
        self.horizontal_velocity_range()
        self.vertical_velocity_range()
        self.ui.comboBox.currentTextChanged.connect(self.handle_selection)
        self.ui.pushButton_18.clicked.connect(self.montecarlo_start)
        self.ui.pushButton_11.clicked.connect(self.select_file_2)
        self.ui.pushButton_17.clicked.connect(self.select_file_3)
        self.ui.pushButton_12.clicked.connect(self.select_file_5)
        self.ui.pushButton_8.clicked.connect(self.select_file_4)
        self.ui.pushButton_26.clicked.connect(self.select_file_6)
        self.ui.pushButton_28.clicked.connect(self.select_file_7)
        self.ui.doubleSpinBox.valueChanged.connect(self.off_centering)
        self.ui.doubleSpinBox_2.valueChanged.connect(self.initial_height)
        self.ui.doubleSpinBox_19.valueChanged.connect(self.horizontal_velocity)
        self.ui.doubleSpinBox_18.valueChanged.connect(self.vertical_velocity)
        self.ui.doubleSpinBox_3.valueChanged.connect(self.pitch_angle)
        self.ui.doubleSpinBox_6.valueChanged.connect(self.wind_strength)
        self.ui.doubleSpinBox_7.valueChanged.connect(self.simulation_number)
        self.ui.doubleSpinBox_8.valueChanged.connect(self.initial_height_range)
        self.ui.doubleSpinBox_9.valueChanged.connect(self.initial_height_range)
        self.ui.doubleSpinBox_20.valueChanged.connect(self.off_centering_range)
        self.ui.doubleSpinBox_21.valueChanged.connect(self.off_centering_range)
        self.ui.doubleSpinBox_16.valueChanged.connect(self.pitch_angle_range)
        self.ui.doubleSpinBox_17.valueChanged.connect(self.pitch_angle_range)
        self.ui.doubleSpinBox_14.valueChanged.connect(self.wind_strength_range)
        self.ui.doubleSpinBox_15.valueChanged.connect(self.wind_strength_range)
        self.ui.doubleSpinBox_10.valueChanged.connect(self.horizontal_velocity_range)
        self.ui.doubleSpinBox_11.valueChanged.connect(self.horizontal_velocity_range)
        self.ui.doubleSpinBox_12.valueChanged.connect(self.vertical_velocity_range)
        self.ui.doubleSpinBox_13.valueChanged.connect(self.vertical_velocity_range)

        self.ui.pushButton_5.clicked.connect(self.select_file)
        self.ui.pushButton_6.clicked.connect(self.analysis_file)
        self.ui.pushButton_7.clicked.connect(self.open_file)
        self.ui.pushButton_13.clicked.connect(lambda: setattr(self, 'choice', 'terminal_x') or self.analysis_file_2())
        self.ui.pushButton_14.clicked.connect(lambda: setattr(self, 'choice', 'adjust_time') or self.analysis_file_2())
        self.ui.pushButton_15.clicked.connect(lambda: setattr(self, 'choice', 'settling_time') or self.analysis_file_2())
        self.ui.pushButton_16.clicked.connect(lambda: setattr(self, 'choice', 'fuel_consumption') or self.analysis_file_2())
        self.ui.pushButton_19.clicked.connect(lambda: setattr(self, 'choice', 'success_rate') or self.analysis_file_2())
        self.ui.pushButton_20.clicked.connect(lambda: setattr(self, 'choice', 'effort') or self.analysis_file_2())
        self.ui.pushButton_21.clicked.connect(lambda: setattr(self, 'choice_2', 'velocity') or self.analysis_file_3())
        self.ui.pushButton_22.clicked.connect(lambda: setattr(self, 'choice_2', 'v_TVC') or self.analysis_file_3())
        self.ui.pushButton_23.clicked.connect(lambda: setattr(self, 'choice_2', 'theta') or self.analysis_file_3())
        self.ui.pushButton_24.clicked.connect(lambda: setattr(self, 'choice_2', 'position') or self.analysis_file_3())
        self.ui.pushButton_25.clicked.connect(lambda: setattr(self, 'choice_2', 'v_fins') or self.analysis_file_3())

        self.file_paths = []
        #target_w, target_h = 500, 570
        #self.inference.env.bg_img = cv2.resize(self.inference.env.bg_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    #page_6
    def select_file_7(self):
        self.two_d = False
        self.one_d = True
        self.four_d = False
        QMessageBox.information(self, "Hint", "Analysis is completed")

    def select_file_6(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*);;Text Files (*.txt)")

        if file_name:   
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            
            if not base_name.endswith("trajectory"):
                QMessageBox.warning(self, "Warning", "Please select a valid trajectory file (filename must end with 'trajectory').")
                return 
            
            self.ui.lineEdit_17.setText(file_name)

    def select_file_2(self):
        # 4. 打开文件对话框
        # 返回值是一个元组 (文件名, 文件类型)
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*);;Text Files (*.txt)")

        # 5. 如果用户选中了文件（没有直接关闭对话框），则更新 LineEdit
        if file_name:
            self.ui.lineEdit_7.setText(file_name)

    def select_file_3(self):
        # 4. 打开文件对话框
        # 返回值是一个元组 (文件名, 文件类型)
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*);;Text Files (*.txt)")

        # 5. 如果用户选中了文件（没有直接关闭对话框），则更新 LineEdit
        if file_name:
            self.ui.lineEdit_8.setText(file_name)

    def select_file_4(self):
        self.file_paths = []  # 清空之前的文件路径
        required_files = 4

        for i in range(required_files):
            file_name, _ = QFileDialog.getOpenFileName(
                self, 
                f"Select the file {i+1}", 
                "", 
                "All Files (*);;Text Files (*.txt)"
            )
            if file_name in self.file_paths:
                QMessageBox.warning(self, "Warning", f"File '{file_name}' has already been selected!")
                continue 
            if not file_name:  # 用户取消选择
                QMessageBox.warning(self, "Warning", "Please select 4 files!")
                return
            self.file_paths.append(file_name)

        self.two_d = False
        self.one_d = False
        self.four_d = True
        QMessageBox.information(self, "Hint", "Analysis is completed!")

    def select_file_5(self):
        self.two_d = True
        self.one_d = False
        self.four_d = False
        QMessageBox.information(self, "Hint", "Analysis is completed")

    def analysis_file_3(self):
        if self.one_d:
            plt.close('all')
            file_path = self.ui.lineEdit_17.text()
            base_name = os.path.basename(file_path)
            if base_name.endswith("_trajectory.csv"):
                result = base_name[:-len("_trajectory.csv")]
            else:
                result = os.path.splitext(base_name)[0]
            df = pd.read_csv(file_path)
            x = df['x']
            y = df['y']
            vphi = df['v_TVC']
            vdelta = df['v_Fins']
            vx = df['vx']
            vy = df['vy']
            theta = df['theta']
            times = df['times']
            if self.choice_2 == 'velocity':
                plt.figure(figsize=(10, 6))
                plt.plot(times, vx, label='vx', linewidth=2)
                plt.plot(times, vy, label='vy', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity (m/s)')
                plt.title('Simulation curve of Rocket Landing')
                plt.legend()
                plt.grid(True)
            elif self.choice_2 == 'position':
                plt.figure(figsize=(10, 6))
                plt.plot(times, x, label='x', linewidth=2)
                plt.plot(times, y, label='y', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Position (m)')
                plt.title('Simulation curve of Rocket Landing')
                plt.legend()
                plt.grid(True)
            elif self.choice_2 == 'v_TVC':
                plt.figure(figsize=(10, 6))
                plt.plot(times, vphi, label='v_TVC', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity (m/s)')
                plt.title('Simulation curve of Rocket Landing')
                plt.legend()
                plt.grid(True)
            elif self.choice_2 == 'v_fins':
                plt.figure(figsize=(10, 6))
                plt.plot(times, vdelta, label='v_Fins', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity (m/s)')
                plt.title('Simulation curve of Rocket Landing')
                plt.legend()
                plt.grid(True)
            elif self.choice_2 == 'theta':
                plt.figure(figsize=(10, 6))
                plt.plot(times, theta, label='theta', linewidth=2)
                plt.xlabel('Time (s)')
                plt.ylabel('Angle (rad)')
                plt.title('Simulation curve of Rocket Landing')
                plt.legend()
                plt.grid(True)
            folder = os.path.join('./', 'simulation_curve_png')
            save_path = os.path.join(folder, f'{result}_{self.choice_2}.png')
            save_path = save_path.replace(".csv", "")
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            # 使用 QPixmap 加载图像
            qpixmap = QPixmap(save_path)

            # 获取 QLabel 的尺寸
            label_size = self.ui.label_59.size()

            # 缩放图像以适应 QLabel 的尺寸（注意参数顺序）
            scaled_pixmap = qpixmap.scaled(
                label_size.width(), 
                label_size.height(), 
                aspectMode=Qt.KeepAspectRatio, 
                mode=Qt.SmoothTransformation
            )

            # 设置缩放后的图像到 QLabel
            self.ui.label_59.setPixmap(scaled_pixmap)
        else:
            QMessageBox.warning(self, "Warning", "Please select 1 file first")
            return

    def analysis_file_2(self):
        if self.two_d:
            plt.close('all')
            file_path_1 = self.ui.lineEdit_7.text()
            file_path_2 = self.ui.lineEdit_8.text()
            if file_path_1 == "" or file_path_2 == "":
                QMessageBox.warning(self, "Warning", "Please select files first")
                return
            file_name_1 = os.path.basename(file_path_1)
            file_name_2 = os.path.basename(file_path_2)

            base_name_1 = os.path.splitext(file_name_1)[0]
            base_name_2 = os.path.splitext(file_name_2)[0]
            meta_path_1 = os.path.join(os.path.dirname(file_path_1), f'{base_name_1}_meta.json')
            meta_path_2 = os.path.join(os.path.dirname(file_path_2), f'{base_name_2}_meta.json')
            with open(meta_path_1, 'r') as f:
                meta_1 = json.load(f)
            with open(meta_path_2, 'r') as f:
                meta_2 = json.load(f)
            scenario_1 = meta_1.get('scenario_name', 'Unknown')
            strategy_1 = meta_1.get('strategy', 'Unknown')
            
            scenario_2 = meta_2.get('scenario_name', 'Unknown')
            strategy_2 = meta_2.get('strategy', 'Unknown')
            # 比较逻辑
            is_scenario_same = (scenario_1 == scenario_2)
            is_strategy_same = (strategy_1 == strategy_2)
            if is_scenario_same and is_strategy_same:
                QMessageBox.warning(self, "Warning", "Please select files with the different scenario or strategy")
                return
            elif is_scenario_same:
                same_name = 'scenario'
                comparison_name = 'strategy'
            elif is_strategy_same:
                same_name = 'strategy'
                comparison_name = 'scenario'
            else:
                QMessageBox.warning(self, "Warning", "Please select files with the different scenario or strategy")
                return

            df_1 = pd.read_csv(file_path_1)
            df_2 = pd.read_csv(file_path_2)
            success_1 = df_1[df_1['success'] == 1]
            success_2 = df_2[df_2['success'] == 1]
            if success_1.empty:
                QMessageBox.warning(self, "Warning", f"No successful trials found in {file_name_1}")
                return
            if success_2.empty:
                QMessageBox.warning(self, "Warning", f"No successful trials found in {file_name_2}")
                return

            # 计算每列的平均值
            mean_1 = success_1[self.choice].mean()
            mean_2 = success_2[self.choice].mean()

            # 绘制柱状图
            fig = plt.figure(figsize=(10, 6))

            # 设置柱状图的位置和宽度
            x_pos = np.arange(2)
            width = 0.6

            # 创建柱状图
            bars = plt.bar(x_pos, [mean_1, mean_2], width, 
                        color=['skyblue', 'lightcoral'], 
                        edgecolor='black', label = [file_name_1, file_name_2], linewidth=1.2)

            # 设置图表标题和标签
            unit_map = {
                'terminal_x': 'm',
                'adjust_time': 's',
                'settling_time': 's',
                'fuel_consumption': 'kg',
                'effort': 'm/s',
                'success_rate': '%' # 如果是百分比
            }
            
            # 获取单位，如果没有定义则默认为空
            unit = unit_map.get(self.choice, '')
            plt.title(f'{self.choice} under the same {same_name}', fontsize=14, fontweight='bold')
            plt.xlabel(f'{comparison_name}', fontsize=12)
            if unit:
                plt.ylabel(f'{self.choice} ({unit})', fontsize=12)
            else:
                plt.ylabel(f'{self.choice}', fontsize=12)
            plt.xticks(x_pos, [file_name_1, file_name_2])

            # 在每个柱子上方显示数值
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

            # 添加网格线使图表更易读
            plt.grid(axis='y', alpha=0.3, linestyle='--')

            # 调整y轴范围，使图表更美观
            plt.ylim(0, max(mean_1, mean_2) * 1.15)
            
            folder = os.path.join('./', '2d_png')
            save_path = os.path.join(folder, f'{file_name_1}_and_{file_name_2}_{self.choice}.png')
            save_path = save_path.replace(".csv", "")
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            # 使用 QPixmap 加载图像
            qpixmap = QPixmap(save_path)

            # 获取 QLabel 的尺寸
            label_size = self.ui.label_59.size()

            # 缩放图像以适应 QLabel 的尺寸（注意参数顺序）
            scaled_pixmap = qpixmap.scaled(
                label_size.width(), 
                label_size.height(), 
                aspectMode=Qt.KeepAspectRatio, 
                mode=Qt.SmoothTransformation
            )

            # 设置缩放后的图像到 QLabel
            self.ui.label_59.setPixmap(scaled_pixmap)
        elif self.four_d:
            plt.close('all')
            # 提取文件名和文件夹名
            file_names = [os.path.basename(path) for path in self.file_paths]
            folders = [os.path.basename(os.path.dirname(path)) for path in self.file_paths]

            # 找出不重复的文件名和文件夹名
            unique_file_names = list(set(file_names))
            unique_folders = list(set(folders))

            if len(unique_folders) != 2:
                QMessageBox.warning(self, "Warning", "Please select files from two different strategy folders")
                return
            # 读取数据并检查成功试验
            dataframes = [pd.read_csv(path) for path in self.file_paths]
            success_data = [df[df['success'] == 1] for df in dataframes]

            # 检查是否有空的成功试验数据
            for i, (name, success_df) in enumerate(zip(file_names, success_data)):
                if success_df.empty:
                    QMessageBox.warning(self, "Warning", f"No successful trials found in {name}")
                    return
            x_labels = unique_folders
            y_labels = unique_file_names

            # 构建均值字典
            means = {}
            for folder in x_labels:
                for file in y_labels:
                    file_path = os.path.join('./', folder, file)
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            mean_value = df[self.choice].mean()
                            means[f"{folder}_{file}"] = mean_value
                        except Exception as e:
                            QMessageBox.warning(self, "Warning", f"Failed to process {file_path}: {e}")
                            return
                    else:
                        QMessageBox.warning(self, "Warning", f"File not found: {file_path}")
                        return

            # 构建 z 轴数据
            z_data = np.array([
                [means.get(f"{x_labels[0]}_{y_labels[0]}", 0), means.get(f"{x_labels[1]}_{y_labels[0]}", 0)],
                [means.get(f"{x_labels[0]}_{y_labels[1]}", 0), means.get(f"{x_labels[1]}_{y_labels[1]}", 0)]
            ])

            # 创建网格
            x_pos = np.arange(len(x_labels))
            y_pos = np.arange(len(y_labels))
            x_pos, y_pos = np.meshgrid(x_pos, y_pos)
            x_pos = x_pos.flatten()
            y_pos = y_pos.flatten()
            z_pos = np.zeros_like(x_pos, dtype=float)

            # 设置柱子的尺寸
            dx = dy = 0.8 * np.ones_like(z_pos)
            dz = z_data.flatten()

            # 创建3D图形
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # 绘制3D柱状图
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 不同条件的颜色
            cs = []
            for i in range(len(y_labels)):
                cs.extend([colors[i]] * len(x_labels))

            ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=cs, alpha=0.8)

            # 设置标签
            ax.set_xlabel('Environment', labelpad=15, fontsize=12)
            ax.set_ylabel('Condition', labelpad=15, fontsize=12)
            ax.set_zlabel('Settling time', labelpad=15, fontsize=12)

            # 设置x轴刻度标签
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels)

            # 设置y轴刻度标签
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)

            # 设置标题
            plt.title(f'Comparison of {self.choice} under different envs and strategies', fontsize=14, fontweight='bold')
            folder = os.path.join('./', '3d_png')
            file_name_str = "_".join(unique_file_names)
            file_name_str = file_name_str.replace(".csv", "")
            save_path = os.path.join(folder, f'{file_name_str}_{self.choice}.png')
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            qpixmap = QPixmap(save_path)

            # 获取 QLabel 的尺寸
            label_size = self.ui.label_59.size()

            # 缩放图像以适应 QLabel 的尺寸（注意参数顺序）
            scaled_pixmap = qpixmap.scaled(
                label_size.width(), 
                label_size.height(), 
                aspectMode=Qt.KeepAspectRatio, 
                mode=Qt.SmoothTransformation
            )

            # 设置缩放后的图像到 QLabel
            self.ui.label_59.setPixmap(scaled_pixmap)
        else:
            QMessageBox.warning(self, "Warning", "Please select files first")
            return

    #page_2
    def off_centering(self):
        self.init_set['off_centering'] = self.ui.doubleSpinBox.value()

    def initial_height(self):
        self.init_set['initial_height'] = self.ui.doubleSpinBox_2.value()

    def horizontal_velocity(self):
        self.init_set['horizontal_velocity'] = self.ui.doubleSpinBox_19.value()

    def vertical_velocity(self):
        self.init_set['vertical_velocity'] = self.ui.doubleSpinBox_18.value()

    def pitch_angle(self):
        self.init_set['pitch_angle'] = np.deg2rad(self.ui.doubleSpinBox_3.value())

    def wind_strength(self):
        self.init_set['wind_strength'] = self.ui.doubleSpinBox_6.value()

    #page_5
    def select_file(self):
        # 4. 打开文件对话框
        # 返回值是一个元组 (文件名, 文件类型)
        file_name, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*);;Text Files (*.txt)")

        # 5. 如果用户选中了文件（没有直接关闭对话框），则更新 LineEdit
        if file_name:
            self.ui.lineEdit.setText(file_name)
    
    def analysis_file(self):
        file_name = self.ui.lineEdit.text()
        if file_name == "":
            QMessageBox.warning(self, "Warning", "Please select a file first")
            return
        
        try:
            file_path = os.path.abspath(file_name)
            
            # --- 1. 读取 JSON 元数据获取范围 ---
            # 假设 JSON 文件名为 CSV 文件名去掉后缀后加上 _meta.json
            base_name = os.path.splitext(file_path)[0]
            json_path = f"{base_name}_meta.json"
            
            state_range = {}
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    # 根据之前保存的结构，范围在 meta_data['state_range'] 中
                    if 'state_range' in meta_data:
                        state_range = meta_data['state_range']
            else:
                # 兼容旧模式：如果找不到 JSON，尝试从 CSV 第一行读取（可选，用于向后兼容）
                # 这里为了简化，如果没找到 JSON 则提示或留空
                print(f"Warning: Metadata file not found at {json_path}")

            # 更新 UI 显示范围
            # 注意：JSON 中保存的 key 可能是 'x', 'y' 等，也可能是 'off_centering' 等，取决于保存时的逻辑
            # 根据 montecarlo_pid.py 的逻辑，PID 模式下保存的是 'x', 'y', 'vx', 'vy', 'theta', 'wind'
            
            def get_range_text(key, unit=""):
                if key in state_range:
                    vals = state_range[key]
                    # 处理列表或元组
                    if isinstance(vals, (list, tuple)) and len(vals) == 2:
                        return f"{vals[0]} {unit}~ {vals[1]} {unit}"
                return "N/A"

            # 映射 UI 控件到对应的 JSON Key
            # 注意：你的 montecarlo_pid.py 中 PID 分支保存的 key 是 x, y, vx, vy, theta, wind
            self.ui.lineEdit_2.setText(get_range_text("x", "m"))       # off_centering / x
            self.ui.lineEdit_3.setText(get_range_text("y", "m"))       # initial_height / y
            self.ui.lineEdit_4.setText(get_range_text("vx", "m/s"))    # horizontal_velocity / vx
            self.ui.lineEdit_5.setText(get_range_text("vy", "m/s"))    # vertical_velocity / vy
            self.ui.lineEdit_6.setText(get_range_text("theta", "°"))   # pitch_angle / theta
            self.ui.lineEdit_15.setText(get_range_text("wind", "m/s")) # wind_strength / wind

            # --- 2. 读取 CSV 数据计算统计值 ---
            df = pd.read_csv(file_path)

            count = (df['success'] == 1).sum()
            total_count = df['success'].notna().sum()
            ratio = count / total_count if total_count > 0 else 0
            self.ui.lineEdit_14.setText(f"{ratio:.2%}")

            success_data = df[df['success'] == 1]
            if success_data.empty:
                QMessageBox.warning(self, "Warning", "No successful trials found")
                # 即使没有成功数据，也显示范围，所以这里不直接 return，而是继续显示其他为 N/A 或 0
                # 如果希望严格报错，可以 return
                # return 
            
            # 计算均值，如果为空则设为 0 或 NaN
            def safe_mean(series):
                if series.empty:
                    return 0.0
                return series.mean()

            terminal_x = safe_mean(success_data['terminal_x'])
            self.ui.lineEdit_10.setText(f"{terminal_x:.2f} m")
            
            adjust_time = safe_mean(success_data['adjust_time'])
            self.ui.lineEdit_13.setText(f"{adjust_time:.2f} s")
            
            settling_time = safe_mean(success_data['settling_time'])
            self.ui.lineEdit_12.setText(f"{settling_time:.2f} s")
            
            fuel_consumption = safe_mean(success_data['fuel_consumption'])
            self.ui.lineEdit_11.setText(f"{fuel_consumption:.2f} kg")
            
            effort = safe_mean(success_data['effort'])
            self.ui.lineEdit_16.setText(f"{effort:.2f} m/s")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误以便调试
            return None
    def open_file(self):
        file_name = self.ui.lineEdit.text()
        file_path = os.path.abspath(file_name)
        os.startfile(file_path)

    #page_4
    def handle_selection(self, text):
        if text == "RL":
            self.ifRL = True
            self.ifPID = False
            self.ifPIDRL = False
        elif text == "PID":
            self.ifPID = True
            self.ifRL = False
            self.ifPIDRL = False
        elif text == "PID+RL":
            self.ifPIDRL = True
            self.ifRL = False
            self.ifPID = False

    def montecarlo_start(self):
        if self.ifRL:
            file_name = self.ui.lineEdit_9.text()
            current_text = self.ui.comboBox.currentText()
            scenario_name = self.ui.lineEdit_18.text()
            folder = os.path.join('./', f'{current_text}_results')
             # 确保目录存在
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            if file_name == "":
                QMessageBox.warning(self, "Warning", "Please input file name")
                return
            elif self.ui.radioButton_14.isChecked() or self.ui.buttonGroup_2.checkedButton() == None:
                if scenario_name == "":
                    QMessageBox.warning(self, "Warning", "Please input self-defined scenario name or select a scenario")
                    return
            if self.ui.radioButton_14.isChecked() != True and self.ui.buttonGroup_2.checkedButton() != None:
                scenario_name = self.ui.buttonGroup_2.checkedButton().text()
            self.file_name = f'{scenario_name}_PID.csv'
            file_page = os.path.join(folder, self.file_name)
            self.state_range = {
                "off_centering": self.off_centering_range_val,
                "initial_height": self.initial_height_range_val,
                "horizontal_velocity": self.horizontal_velocity_range_val,
                "vertical_velocity": self.vertical_velocity_range_val,
                "pitch_angle": self.pitch_angle_range_val,
                "wind_strength": self.wind_strength_range_val,
            }
            self.analysis.run_batch(self.number, file_page, self.state_range)
        elif self.ifPID:
            scenario_name = self.ui.lineEdit_18.text()
            current_text = self.ui.comboBox.currentText()
            folder = os.path.join('./', f'{current_text}_results')
             # 确保目录存在
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            if self.ui.radioButton_14.isChecked() or self.ui.buttonGroup_2.checkedButton() == None:
                if scenario_name == "":
                    QMessageBox.warning(self, "Warning", "Please input self-defined scenario name or select a scenario")
                    return
            if self.ui.radioButton_14.isChecked() != True and self.ui.buttonGroup_2.checkedButton() != None:
                scenario_name = self.ui.buttonGroup_2.checkedButton().text()
                if scenario_name == "Center point":
                    self.seed = 1
                elif scenario_name == "Left offset":
                    self.seed = 2
                elif scenario_name == "Right offset":
                    self.seed = 3
                elif scenario_name == "Strong crosswind":
                    self.seed = 4
                elif scenario_name == "High-speed descent":
                    self.seed = 5
                elif scenario_name == "Large angle tilt":
                    self.seed = 6
                elif scenario_name == "Extreme conditions":
                    self.seed = 7
            self.file_name = f'{scenario_name}_PID.csv'
            self.ui.lineEdit_9.setText(self.file_name)
            file_page = os.path.join(folder, self.file_name)
            self.state_range = {
                "x": self.off_centering_range_val,
                "y": self.initial_height_range_val,
                "vx": self.horizontal_velocity_range_val,
                "vy": self.vertical_velocity_range_val,
                "theta": self.pitch_angle_range_val,
                "wind": self.wind_strength_range_val,
            }
            self.analysis_pid.run_batch(int(self.number), file_page, self.state_range, self.seed, scenario_name)


    def simulation_number(self):
        self.number = self.ui.doubleSpinBox_7.value()
    
    def vertical_velocity_range(self):
        minimum = self.ui.doubleSpinBox_12.value()
        maximum = self.ui.doubleSpinBox_13.value()
        if minimum > maximum:
            QMessageBox.warning(self, "Warning", "incorrect order of vertical velocity")
            return
        self.vertical_velocity_range_val = (minimum, maximum)

    def off_centering_range(self):
        minimum = self.ui.doubleSpinBox_20.value()
        maximum = self.ui.doubleSpinBox_21.value()
        if minimum > maximum:
            QMessageBox.warning(self, "Warning", "incorrect order of off-centering")
            return
        self.off_centering_range_val = (minimum, maximum)

    def initial_height_range(self):
        minimum = self.ui.doubleSpinBox_8.value()
        maximum = self.ui.doubleSpinBox_9.value()
        if minimum > maximum:
            QMessageBox.warning(self, "Warning", "incorrect order of initial height")
            return
        self.initial_height_range_val = (minimum, maximum)
    def pitch_angle_range(self):
        minimum = self.ui.doubleSpinBox_16.value()
        maximum = self.ui.doubleSpinBox_17.value()
        if minimum > maximum:
            QMessageBox.warning(self, "Warning", "incorrect order of pitch angle")
            return
        self.pitch_angle_range_val = (minimum, maximum)

    def wind_strength_range(self):
        minimum = self.ui.doubleSpinBox_14.value()
        maximum = self.ui.doubleSpinBox_15.value()
        if minimum > maximum:
            QMessageBox.warning(self, "Warning", "incorrect order of wind strength")
            return
        self.wind_strength_range_val = (minimum, maximum)

    def horizontal_velocity_range(self):
        minimum = self.ui.doubleSpinBox_10.value()
        maximum = self.ui.doubleSpinBox_11.value()
        if minimum > maximum:
            QMessageBox.warning(self, "Warning", "incorrect order of horizontal velocity")
            return
        self.horizontal_velocity_range_val = (minimum, maximum)

    #page_3
    def draw_trajectory(self, canvas, color=(255, 0, 0)):
        env = self.inference_pid.env
        pannel_w, pannel_h = 130, 150
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        try:
            sw, sh = pannel_w / env.viewport_w, pannel_h / env.viewport_h  # scale factors

            # draw horizon line
            range_x, range_y = env.world_x_max - env.world_x_min, env.world_y_max - env.world_y_min
            pts = [[env.world_x_min + range_x/3, env.H/2], [env.world_x_max - range_x/3, env.H/2]]
            pts_px = env.wd2pxl(pts)
            x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
            x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                     color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

            # draw vertical line
            pts = [[0, env.H/2], [0, env.H/2+range_y/20]]
            pts_px = env.wd2pxl(pts)
            x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
            x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                     color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

            if len(self.state_buffer) >= 2:
                # draw traj
                pts = []
                for state in self.state_buffer:
                    pts.append([state['x'], state['y']])
                pts_px = env.wd2pxl(pts)

                dn = 5
                for i in range(0, len(pts_px)-dn, dn):
                    x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
                    x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)
                    cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        except AttributeError:
            # 防止环境变量尚未初始化完全的情况
            pass

        # 动态获取当前画布宽高，安全叠加到右上角
        canvas_h, canvas_w = canvas.shape[:2]
        if canvas_w > pannel_w + 0 and canvas_h > pannel_h + 0:
            roi_x1, roi_x2 = canvas_w - 0 - pannel_w, canvas_w - 0
            roi_y1, roi_y2 = 0, 0 + pannel_h
            
            # 使用 astype 显式转换防止数据类型引起 QImage 崩溃
            canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = (0.6 * canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4 * traj_pannel).astype(np.uint8)

        return canvas
    def start_simulation(self):
        # 1. 校验策略选择
        if not self.ui.radioButton_3.isChecked() and not self.ui.radioButton_4.isChecked() and not self.ui.radioButton_5.isChecked():
            QMessageBox.warning(self, "Warning", "Please select a strategy first")
            return

        # 2. 根据选择加载模型
        if self.ui.radioButton_3.isChecked():
            if self.ui.radioButton.isChecked():
                self.init_set = None
            self.state = self.inference_RL.env.reset(self.init_set)
            self.step_count = 0
            self.state_buffer = [] 
            self.state_buffer.append({'x': self.inference_RL.env.state['x'], 'y': self.inference_RL.env.state['y']})
            self.timer.start(30) # 30毫秒刷新一次
        elif self.ui.radioButton_4.isChecked():
            if self.ui.buttonGroup.checkedButton() and self.ui.radioButton_90.isChecked() != True:
                self.init_set = None
                self.seed = None
            if self.ui.radioButton.isChecked():
                self.seed = 1
            elif self.ui.radioButton_2.isChecked():
                self.seed = 2
            elif self.ui.radioButton_6.isChecked():
                self.seed = 3
            elif self.ui.radioButton_7.isChecked():
                self.seed = 4
            elif self.ui.radioButton_9.isChecked():
                self.seed = 5
            elif self.ui.radioButton_8.isChecked():
                self.seed = 6
            elif self.ui.radioButton_10.isChecked():
                self.seed = 7
            self.state = self.inference_pid.env.reset(self.init_set, seed=self.seed)
            self.step_count = 0
            self.state_buffer = [] 
            self.state_buffer.append({'x': self.inference_pid.env.state['x'], 'y': self.inference_pid.env.state['y']})
            self.timer.start(30) # 30毫秒刷新一次

        elif self.ui.radioButton_5.isChecked():
            if self.ui.buttonGroup.checkedButton() and self.ui.radioButton_90.isChecked() != True:
                self.init_set = None
                self.seed = None
            if self.ui.radioButton.isChecked():
                self.seed = 1
            elif self.ui.radioButton_2.isChecked():
                self.seed = 2
            elif self.ui.radioButton_6.isChecked():
                self.seed = 3
            elif self.ui.radioButton_7.isChecked():
                self.seed = 4
            elif self.ui.radioButton_9.isChecked():
                self.seed = 5
            elif self.ui.radioButton_8.isChecked():
                self.seed = 6
            elif self.ui.radioButton_10.isChecked():
                self.seed = 7
            self.state = self.inference_pid_RL.env.reset(self.init_set, seed=self.seed)
            self.step_count = 0
            self.state_buffer = [] 
            self.state_buffer.append({'x': self.inference_pid_RL.env.state['x'], 'y': self.inference_pid_RL.env.state['y']})
            self.timer.start(30) # 30毫秒刷新一次

    def pause_simulation(self):
        """暂停仿真"""
        if self.ui.radioButton_3.isChecked() or self.ui.radioButton_4.isChecked() or self.ui.radioButton_5.isChecked():
            if self.timer.isActive():
                self.timer.stop()

    def continue_simulation(self):
        """继续仿真"""
        if self.ui.radioButton_3.isChecked() or self.ui.radioButton_4.isChecked() or self.ui.radioButton_5.isChecked():
            if not self.timer.isActive():
                self.timer.start(30)

    def abort_simulation(self):
        """中止并重置"""
        if self.ui.radioButton_3.isChecked() or self.ui.radioButton_4.isChecked() or self.ui.radioButton_5.isChecked():
            self.timer.stop()
            if self.ui.radioButton_3.isChecked():
                self.state = self.inference_RL.env.reset(self.init_set) # 彻底重置状态
                self.state_buffer = [] # 清空状态缓冲区
                # 渲染一帧初始画面（不带火焰），让界面回到初始位置
                frame_0, _ = self.inference_RL.env.render()
                self.update_frame(frame_0)
                self.ui.pushButton.setEnabled(True)
            elif self.ui.radioButton_4.isChecked():
                self.state = self.inference_pid.env.reset(self.init_set, self.seed) # 彻底重置状态
                self.state_buffer = [] # 清空状态缓冲区
                # 渲染一帧初始画面（不带火焰），让界面回到初始位置
                frame_0, _ = self.inference_pid.env.render()
                self.update_frame(frame_0)
                self.ui.pushButton.setEnabled(True)
            elif self.ui.radioButton_5.isChecked():
                self.state = self.inference_pid_RL.env.reset(self.init_set, self.seed) # 彻底重置状态
                self.state_buffer = [] # 清空状态缓冲区
                # 渲染一帧初始画面（不带火焰），让界面回到初始位置
                frame_0, _ = self.inference_pid_RL.env.render()
                self.update_frame(frame_0)
                self.ui.pushButton.setEnabled(True)
    
    def update_frame(self, frame):
        # 1. 获取 label_71 的当前实际像素宽高
        label_w = self.ui.label_71.width()
        label_h = self.ui.label_71.height()
        
        # 2. 先把原始主画面强制拉伸铺满整个 label
        # 这样主画面就适配了 label_71 的宽高
        frame_resized = cv2.resize(frame, (label_w, label_h), interpolation=cv2.INTER_LINEAR)
        
        # 3. 在拉伸后的主画面上，叠加轨迹画板
        # 因为底图已经是最终尺寸，画板按照 200x180 的像素盖上去，绝对不会再变形
        frame_with_traj = self.draw_trajectory(frame_resized)
        
        # 4. 格式转换: BGR -> RGB (如果是RGB环境请保持 Format_RGB888)
        height, width, channel = frame_with_traj.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_with_traj.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 5. 直接显示到 Label 上
        # 因为我们在第2步已经把尺寸调整得和 label_71 严丝合缝，这里直接放上去就行
        self.ui.label_71.setPixmap(QPixmap.fromImage(q_img))       

    def run_one_step(self):
        if self.step_count < self.max_steps:
            # --- 渲染逻辑 (获取两帧) ---
            if self.ui.radioButton_3.isChecked():
                frame_0, frame_1 ,state, done= self.inference_RL.run_inference(self.state, self.init_interference)
                self.state = state
                
                self.state_buffer.append({'x': self.inference_RL.env.state['x'], 'y': self.inference_RL.env.state['y']})
                # --- 交替显示以实现闪烁 ---
                current_frame = frame_1 if self.show_flame else frame_0
                self.show_flame = not self.show_flame
                
                # --- 更新 UI ---
                self.update_frame(current_frame)
                self.update_text_labels()
                
                self.step_count += 1
                if done:
                    self.timer.stop()
            elif self.ui.radioButton_4.isChecked():
                frame_0, frame_1, state, done = self.inference_pid.run_inference()        
                self.state_buffer.append({'x': state['x'], 'y': state['y']})
                # --- 交替显示以实现闪烁 ---
                current_frame = frame_1 if self.show_flame else frame_0
                self.show_flame = not self.show_flame
                
                # --- 更新 UI ---
                self.update_frame(current_frame)
                self.update_text_labels()
                
                self.step_count += 1
                if done:
                    self.timer.stop()
            elif self.ui.radioButton_5.isChecked():
                frame_0, frame_1, state, done = self.inference_pid_RL.run_inference()        
                self.state_buffer.append({'x': state['x'], 'y': state['y']})
                # --- 交替显示以实现闪烁 ---
                current_frame = frame_1 if self.show_flame else frame_0
                self.show_flame = not self.show_flame
                
                # --- 更新 UI ---
                self.update_frame(current_frame)
                self.update_text_labels()
                
                self.step_count += 1
                if done:
                    self.timer.stop()

        else:
            self.timer.stop()

    #page_1
    def update_text_labels(self):
        if self.ui.radioButton_4.isChecked():
            x = self.inference_pid.env.state['x']
            y = self.inference_pid.env.state['y']
            vx = self.inference_pid.env.state['vx']
            vy = self.inference_pid.env.state['vy']
            theta = self.inference_pid.env.state['theta'] * 180 / np.pi  # 弧度转角度显示
            vtheta = self.inference_pid.env.state['vtheta'] * 180 / np.pi
            fuel_consumed = self.inference_pid.env.max_fuel - self.inference_pid.env.fuel # 计算已消耗燃料
            wind_v = self.inference_pid.env.wind_strength
        elif self.ui.radioButton_5.isChecked():
            x = self.inference_pid_RL.env.state['x']
            y = self.inference_pid_RL.env.state['y']
            vx = self.inference_pid_RL.env.state['vx']
            vy = self.inference_pid_RL.env.state['vy']
            theta = self.inference_pid_RL.env.state['theta'] * 180 / np.pi  # 弧度转角度显示
            vtheta = self.inference_pid_RL.env.state['vtheta'] * 180 / np.pi
            fuel_consumed = self.inference_pid_RL.env.max_fuel - self.inference_pid_RL.env.fuel # 计算已消耗燃料
            wind_v = self.inference_pid_RL.env.wind_strength

        # 使用 display() 方法更新 LCD 控件
        # 假设你在 Designer 中命名的控件如下：
        self.ui.lcdNumber.display(f"{x:.2f}")
        self.ui.lcdNumber_2.display(f"{y:.2f}")
        self.ui.lcdNumber_3.display(f"{vx:.2f}")
        self.ui.lcdNumber_4.display(f"{vy:.2f}")
        self.ui.lcdNumber_6.display(f"{vtheta:.1f}")
        self.ui.lcdNumber_5.display(f"{theta:.1f}")
        self.ui.lcdNumber_7.display(int(fuel_consumed)) # 燃料通常显示整数
        self.ui.lcdNumber_8.display(f"{wind_v:.1f}")

    def setup_navigation(self):
        for list_widget, base_index in self.lists:
            # 当列表项改变时切换页面
            list_widget.currentRowChanged.connect(
                lambda row, base=base_index, lw=list_widget: self.on_list_item_clicked(row, base, lw)
            )

    def on_list_item_clicked(self, row, base_index, current_list):
        if row == -1:
            return  # 处理清除选中时的信号

        # 计算并切换到目标页面
        target_index = base_index + row
        self.stacked_widget.setCurrentIndex(target_index)

        # 互斥选中：清除其他列表的选中状态
        for list_widget, _ in self.lists:
            if list_widget != current_list:
                list_widget.blockSignals(True)  # 暂时阻塞信号，防止死循环
                list_widget.setCurrentRow(-1)
                list_widget.blockSignals(False)

    def go_to_homepage(self):
        """设置默认页面为第1页 (索引0)"""
        self.ui.listWidget.setCurrentRow(0)
        self.stacked_widget.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 显示启动图
    pixmap = QPixmap("splash.png")
    splash = QSplashScreen(pixmap)
    splash.show()

    window = RocketApp()
    window.show() # 显示加载的 UI 对象
    splash.finish(window) # 主窗口显示后关闭启动图
    sys.exit(app.exec())