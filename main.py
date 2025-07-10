from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import (QFileDialog, QGraphicsScene, QGraphicsRectItem, QGraphicsPathItem, QApplication,
                             QGraphicsView,QTableWidgetItem,QGraphicsItem,QGraphicsSimpleTextItem,QMessageBox, QGraphicsEllipseItem)
from PyQt5.QtCore import Qt, QRectF, QEvent,QPointF,QSizeF,QThread, pyqtSignal,QTimer
from PyQt5.QtGui import QPen, QColor,QPainterPath,QBrush,QIcon, QPixmap,QPainterPathStroker,QFont
from PyQt5.QtSvg import QGraphicsSvgItem,QSvgRenderer
import fitz
import edr2
import math
import table
from untitled import Ui_MainWindow
import re
from svgpathtools import svg2paths
import threading
from pathlib import Path
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import date


class HandleItem(QGraphicsEllipseItem):
    def __init__(self, rect, parent, role):
        super().__init__(rect, parent)
        self.setBrush(QBrush(Qt.white))
        self.setPen(QPen(Qt.gray, 1))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations,False)
        self.setCursor(Qt.SizeFDiagCursor) # 设置鼠标悬停时的光标样式
        self.role = role  # 'rotate' or 'scale'
        self.setZValue(1)

class ResizableRotatableRectItem(QGraphicsItem):
    def __init__(self, rect=QRectF(0, 0, 120, 80),angle = 0.0, index = None,callback = None, parent=None):
        super().__init__(parent)
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges |
            QGraphicsItem.ItemIsFocusable
        ) # 设置一组标志，控制他的行为和交互特性。 QGraphicsItem.ItemIsMovable 允许这个图元被鼠标拖动移动。 | QGraphicsItem.ItemIsSelectable 允许这个图元被选中，通常被选中后会有边框高亮等视觉反馈
        # QGraphicsItem.ItemSendsGeometryChanges 使图元在位置、大小或变换改变时，会通知相关事件，方便你监听和响应几何变化。 | QGraphicsItem.ItemIsFocusable 允许图元接受键盘焦点，便于接收键盘事件。
        self.setAcceptHoverEvents(True) # 启用悬停事件
        self.index = index
        self.callback = callback  # 回调函数
        self.rect = rect
        self.setRotation(angle)
        self.setTransformOriginPoint(self.rect.center()) # 设置旋转中心为矩形中心
        # self.setTransformOriginPoint(QPointF(0, 0))

        self.border = QGraphicsRectItem(rect, self)
        self.border.setPen(QPen(Qt.red, 2))
        self.border.setBrush(QBrush(Qt.transparent))

        self.rotate_handle = HandleItem(QRectF(0, 0, 20, 20), self, 'rotate')
        self.rotate_handle.setBrush(Qt.gray)
        self.rotate_handle.setCursor(Qt.ClosedHandCursor)
        self.handles = []
        self.init_scale_handles()

        self.is_rotating = False
        self.rot_origin = QPointF()
        self.is_scaling = False
        self.scaling_handle = None
        self.scale_origin = QPointF()

        self.setSelected(False)  # 初始不选中
        self.update_handle_visibility()
        self.activate_flag = False # 记录选中状态

    def init_scale_handles(self):
        size = 8
        for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            handle = HandleItem(QRectF(0, 0, size, size), self, 'scale')
            if dx == 0 and dy == 0:
                handle.setBrush(Qt.red)
            else:
                handle.setBrush(Qt.white)
            if dx != dy:
                handle.setCursor(Qt.SizeBDiagCursor)
            self.handles.append(handle)
        self.update_handles()

    def update_handles(self):
        rect = self.rect
        positions = [
            rect.topLeft(), rect.topRight(),
            rect.bottomRight(), rect.bottomLeft()
        ]
        for handle, pos in zip(self.handles, positions):
            handle.setPos(pos.x() - 4, pos.y() - 4)
        self.rotate_handle.setPos(rect.center().x(), rect.top() - 20)

        # ✅ 将旋转手柄放在矩形中心
        center = rect.center()
        self.rotate_handle.setPos(center.x()-10, center.y()-10)  # 居中（根据手柄大小调整）

    def boundingRect(self):
        return self.rect.adjusted(-40, -40, 40, 40)

    def paint(self, painter, option, widget):
        pass  # 已通过子项绘制

    def mousePressEvent(self, event):
        if self.callback:
            self.callback(self.index) # 传递当前索引
        pos = event.pos()
        for handle in self.handles:
            mapped_pos = handle.mapFromItem(self, pos)
            if handle.contains(mapped_pos):
                self.is_scaling = True
                self.scaling_handle = handle
                self.scale_origin = event.scenePos()
                return
        # mapped_pos = self.rotate_handle.mapFromItem(self, pos)
        # if self.rotate_handle.contains(mapped_pos):
        #     print('旋转')
        #     self.is_rotating = True
        #     self.rot_origin = event.scenePos()
        #     print(self.rot_origin)
        #     return
        scene_pos = event.scenePos()
        handle_scene_rect = self.rotate_handle.mapToScene(self.rotate_handle.boundingRect()).boundingRect()

        # 检查点击点是否在旋转手柄的场景区域内
        if handle_scene_rect.contains(scene_pos):
            # print('旋转')
            self.is_rotating = True
            self.rot_origin = scene_pos
            # print(self.rot_origin)
            return

        super().mousePressEvent(event)


    def mouseMoveEvent(self, event):
        if self.is_rotating:
            center = self.mapToScene(self.rect.center())
            v1 = self.rot_origin - center
            v2 = event.scenePos() - center

            # 使用QVector2D的内置方法计算角度（自动处理方向）
            angle = math.degrees(math.atan2(v2.y(), v2.x()) - math.atan2(v1.y(), v1.x()))

            # 应用旋转（限制单次增量在[-180,180]范围内避免跳跃）
            self.setRotation(self.rotation() + angle)
            self.rot_origin = event.scenePos()  # 更新参考点
            self.update_handles()
            # self.rot_origin = event.scenePos()
        elif self.is_scaling and self.scaling_handle:
            local_origin = self.mapFromScene(self.scale_origin)
            local_now = self.mapFromScene(event.scenePos())
            diff = local_now - local_origin

            index = self.handles.index(self.scaling_handle)
            if index == 0:  # Top-left
                self.rect.setTopLeft(self.rect.topLeft())
            elif index == 1:  # Top-right
                self.rect.setTopRight(self.rect.topRight() + diff)
            elif index == 2:  # Bottom-right
                self.rect.setBottomRight(self.rect.bottomRight() + diff)
            elif index == 3:  # Bottom-left
                self.rect.setBottomLeft(self.rect.bottomLeft())

            self.border.setRect(self.rect)
            self.update_handles()
            self.setTransformOriginPoint(self.rect.center())
            self.scale_origin = event.scenePos()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_rotating = False
        self.is_scaling = False
        self.scaling_handle = None
        super().mouseReleaseEvent(event)

    def update_handle_visibility(self):
        visible = self.isSelected()
        for handle in self.handles:
            handle.setVisible(visible)
        self.rotate_handle.setVisible(visible)

    def shape(self):
        path = QPainterPath()

        # 添加边框矩形区域（作为点击区域）
        path.addRect(self.rect)

        # 添加所有手柄区域
        for handle in self.handles + [self.rotate_handle]:
            # 将手柄的形状从手柄局部坐标映射到当前图元坐标
            handle_path = QPainterPath()
            handle_path.addRect(handle.boundingRect())
            mapped_path = handle.mapToParent(handle_path).boundingRect()
            path.addRect(mapped_path)

        # ✅ 使用描边器扩大点击区域
        stroker = QPainterPathStroker()
        stroker.setWidth(5)  # 可调节范围（单位：像素）
        return stroker.createStroke(path).united(path)

    # 添加 itemChange 覆盖
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            QTimer.singleShot(0, self.update_handle_visibility)
        return super().itemChange(change, value)

class SvgOcrThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, file_path, max_length_val, str_length_val,  region, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.max_length_val = max_length_val
        self.str_length_val = str_length_val
        self.region = region  # (xmin, xmax, ymin, ymax)
        self._is_running = True  # 添加标志

        self._lock = threading.Lock()
        self._completed_task_count = 0
        self._total_task_count = 0

    def stop(self):
        self._is_running = False  # 调用后可中断处理过程


    def run(self):
        try:
            if not self._is_running:
                self.progress.emit(0)
                return
            paths, attributes = svg2paths(self.file_path)
            xmin = self.region[0]
            xmax = self.region[1]
            ymin = self.region[2]
            ymax = self.region[3]

            # 筛选区域
            filtered_paths = []
            for path, attr in zip(paths, attributes):
                transform_str = attr.get('transform', '')
                if edr2.is_path_in_region(path, attr, xmin, xmax, ymin, ymax):
                    filtered_paths.append((path, attr))

            all_x = []
            all_y = []
            for path,attr in filtered_paths:
                for segment in path:
                    # 获取线段上的所有点（起点、终点等）
                    all_x.append(segment.start.real)
                    all_y.append(segment.start.imag)
                    all_x.append(segment.end.real)
                    all_y.append(segment.end.imag)

            if all_x and all_y:
                min_x = min(all_x)
                min_y = min(all_y)
                max_x = max(all_x)
                max_y = max(all_y)
            min_x, max_x, min_y, max_y = edr2.get_transform_coordinate((min_x, max_x, min_y, max_y), attr)

            
            filtered_paths1 = edr2.filter_remain_continuous_paths(filtered_paths)
            groups = edr2.group_paths_by_stroke_width(filtered_paths1)

            self._completed_task_count = 0


            def progress_callback(increment=1):
                with self._lock:
                    self._completed_task_count += increment
                    progress = (self._completed_task_count / self._total_task_count) * 100.0
                self.progress.emit(progress)

            def stop_flag():
                return self._is_running
            # result = edr2.create_images_ocr(result, stop_flag=stop_flag)
            result = edr2.parallel_ocr_by_stroke_width(self, groups, progress_callback=progress_callback, stop_flag=stop_flag)


            if self._is_running:
                # self.progress.emit(100)
                self.finished.emit((result, filtered_paths,min_x, max_x, min_y, max_y))

        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))

class ResizableRectItem(QGraphicsRectItem):
    HANDLE_SIZE = 6

    def __init__(self, rect, parent=None,label_text = ''):
        super().__init__(rect, parent)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.handlePressed = None  # 当前操作的控制点
        self.handles = {}  # 控制点位置
        self.updateHandles()

        # 添加文本标签
        self.label = QGraphicsSimpleTextItem(label_text, self)
        # 设置文本颜色为红色
        self.label.setBrush(QBrush(QColor('red')))
        self.updateLabelPosition()

    def updateHandles(self):
        """更新控制点位置"""
        r = self.rect()
        s = self.HANDLE_SIZE
        self.handles = {
            'top_left': QRectF(r.topLeft() - QPointF(s/2, s/2), QSizeF(s, s)),
            'top_right': QRectF(r.topRight() - QPointF(s/2, s/2), QSizeF(s, s)),
            'bottom_left': QRectF(r.bottomLeft() - QPointF(s/2, s/2), QSizeF(s, s)),
            'bottom_right': QRectF(r.bottomRight() - QPointF(s/2, s/2), QSizeF(s, s)),
        }

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.isSelected():
            pen = QPen(QColor("blue"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(QColor("blue"))
            for rect in self.handles.values():
                painter.drawRect(rect)

    def hoverMoveEvent(self, event):
        """根据鼠标位置改变光标形状"""
        for key, handle in self.handles.items():
            if handle.contains(event.pos()):
                if 'left' in key:
                    self.setCursor(Qt.SizeHorCursor)
                elif 'right' in key:
                    self.setCursor(Qt.SizeHorCursor)
                elif 'top' in key:
                    self.setCursor(Qt.SizeVerCursor)
                elif 'bottom' in key:
                    self.setCursor(Qt.SizeVerCursor)
                return
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        """按下时记录是否点击了控制点"""
        for key, handle in self.handles.items():
            if handle.contains(event.pos()):
                self.handlePressed = key
                break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """拖动控制点改变大小"""
        if self.handlePressed:
            pos = event.pos()
            rect = self.rect()
            if self.handlePressed == 'bottom_right':
                rect.setBottomRight(pos)
            elif self.handlePressed == 'top_left':
                rect.setTopLeft(pos)
            elif self.handlePressed == 'top_right':
                rect.setTopRight(pos)
            elif self.handlePressed == 'bottom_left':
                rect.setBottomLeft(pos)

            self.setRect(rect.normalized())
            self.updateHandles()
            self.updateLabelPosition()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.handlePressed = None
        super().mouseReleaseEvent(event)

    def updateLabelPosition(self):
        """将标签放置在矩形框上方中间"""
        rect = self.rect()
        label_rect = self.label.boundingRect()
        x = rect.center().x() - label_rect.width() / 2 - 3*rect.width()/10
        y = rect.top() - label_rect.height() - 5  # 稍微上移一点
        self.label.setPos(x, y)


    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.updateLabelPosition()
        return super().itemChange(change, value)

class StrInfo:
    def __init__(self, num=None, path=None, text=None, score=None, startPoint=None, width=None, height=None,
                 angle=None,rect = None):
        self.text = text
        self.path = path
        self.score = score
        self.num = num
        self.startPoint = startPoint
        self.width = width
        self.height = height
        self.angle = angle
        self.rect = rect

class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()

        # 初始化：
        self.setupUi(self)
        # 表格初始化
        self.init_table()
        # 创建画布
        self.svg_scene = QGraphicsScene()
        self.svgView.setScene(self.svg_scene)
        self.rect_scene = QGraphicsScene()
        self.rectView.setScene(self.rect_scene)
        # 监控选中/取消图元
        self.rect_scene.selectionChanged.connect(self.on_selection_changed)
        # 绑定按钮点击事件
        self.open_button.clicked.connect(self.load_file)
        self.rect_button.clicked.connect(self.enable_rect_selection)
        self.add_button.clicked.connect(self.add_rect)
        self.det_button.clicked.connect(self.svg_ocr)
        self.replace_button.clicked.connect(self.replace_data)
        # self.gen_button.clicked.connect(self.create_gen)
        self.export_Button.clicked.connect(self.create_gen)

        font = QFont()
        font.setPointSize(14)
        self.textEdit.setFont(font)

        # 设置按钮状态
        self.rect_button.setEnabled(False)
        self.det_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.export_Button.setEnabled(False)
        self.replace_button.setEnabled(False)
        self.export_Button.setEnabled(False)
        # self.gen_button.setEnabled(False)

        self.filepath = None  # pdf文件路径
        self.filepath_svg = None  # svg文件路径
        self.is_clear = False # 清空标志
        self.rect_item = None # 识别区域矩形框
        self.flag_add = False  # 添加识别框标志位
        self.flag_rect = False # 绘制识别区域矩形框标志位
        self.temp_rect_item = None # 框选识别区域的临时矩形框
        self.start_pos = None # 框选识别区域的起点
        self.rect_items = [] # 识别框列表
        self.ell_items = [] # 起始点列表
        self.current_rotation = None # 修改后的识别框角度
        self.filename = None # 文件名

        # 框坐标
        self.min_x = None
        self.max_x = None
        self.min_x = None
        self.max_x = None
        self.length = None
        self.width = None

        self.max_length.setText('500')
        self.str_length.setText('10')


        # pdf中上下表格信息提取的框
        self.rectItem0 = []
        self.rectItem1 = []
        self.rectItem2 = []
        self.rectItem3 = []
        self.rectItem4 = []
        self.rectItem5 = []
        self.rectItem6 = []
        self.rectItem7 = []

        # 坐标变换系数
        self.tran_x = None
        self.tran_y = None

        # 安装事件过滤器，监听鼠标点击
        self.rectView.viewport().installEventFilter(self)
        self.svgView.viewport().installEventFilter(self)

        icon = QIcon(QPixmap(edr2.resource_path("favicon.ico")))
        self.setWindowIcon(icon)
        self.setWindowTitle("工程图识别")

    # 初始化表格
    def init_table(self):
        # 初始化 工程图下表格识别的内容
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["序号", "零件编码", "尺寸"])
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 120)
        self.table.horizontalHeader().setStretchLastSection(True)

        # 初始化 工程图中部识别信息表格
        # 设置表格为可编辑、自适应宽度等属性
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # 创建模型并关联到tableView
        self.model = QtGui.QStandardItemModel()
        self.tableView.setModel(self.model)
        # 设置表头（示例）
        self.model.setHorizontalHeaderLabels(['文本', '位置', '角度', '置信度', '操作'])
        self.tableView.clicked.connect(self.handle_table_click)

        # 初始化 替换表格
        self.repalce_table.setColumnCount(4)
        self.repalce_table.setHorizontalHeaderLabels(["文本", "位置", "角度", "字号"])
        self.repalce_table.setColumnWidth(0, 120)
        self.repalce_table.setColumnWidth(1, 120)
        self.repalce_table.setColumnWidth(2, 120)
        self.repalce_table.setColumnWidth(3, 120)
        self.repalce_table.horizontalHeader().setStretchLastSection(True)


    def eventFilter(self, obj, event):
        if obj in [self.svgView.viewport(), self.rectView.viewport()]:
            if self._tool_mouse(obj.parentWidget(), event):
                return True
            if obj == self.svgView.viewport():
                return self._handle_rect_event(self.svgView, event, pos=True, flag=self.flag_rect)
            # else:
            #     return self.add_resizable_rotatable_rect(self.rectView, event,flag=self.flag_add)
        return super().eventFilter(obj, event)

    def _tool_mouse(self, view, event):
        '''
        鼠标工具：ctrl+滚轮：缩放视图，右键：拖拽视图
        :param view:
        :param event:
        :return:
        '''
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            self.right_button_pressed = True
            self.last_mouse_pos = event.pos()
            return True
        elif event.type() == QEvent.MouseMove and getattr(self, 'right_button_pressed', False):
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() - delta.x())
            view.verticalScrollBar().setValue(view.verticalScrollBar().value() - delta.y())
            return True
        elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
            self.right_button_pressed = False
            return True
        elif event.type() == QEvent.Wheel and QApplication.keyboardModifiers() == Qt.ControlModifier:
            view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            zoom_in_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            view.scale(zoom_in_factor, zoom_in_factor)
            return True
        return False

    def _handle_rect_event(self, view, event, pos=False,addrect = False,flag = None):
        scene = view.scene()
        if flag and event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.start_pos = view.mapToScene(event.pos())
            if self.temp_rect_item and self.temp_rect_item.scene():
                self.temp_rect_item.scene().removeItem(self.temp_rect_item)

            self.temp_rect_item = QGraphicsRectItem()
            self.temp_rect_item.setPen(QPen(QColor(255, 0, 0), 1, Qt.DashLine))
            scene.addItem(self.temp_rect_item)
            return True
        elif flag and event.type() == QEvent.MouseMove and self.start_pos:
            current_pos = view.mapToScene(event.pos())
            rect = QRectF(self.start_pos, current_pos).normalized()
            self.temp_rect_item.setRect(rect)
            return True
        elif flag and event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            end_pos = view.mapToScene(event.pos())
            final_rect = QRectF(self.start_pos, end_pos).normalized()
            # print(f"框选区域: 左上({final_rect.topLeft()}), 右下({final_rect.bottomRight()})")

            if pos:
                self.canvas_xmin = final_rect.topLeft().x()
                self.canvas_ymin = final_rect.topLeft().y()
                self.canvas_xmax = final_rect.bottomRight().x()
                self.canvas_ymax = final_rect.bottomRight().y()
                self.flag_rect = False
                self.flag_add = False
            if self.rect_item and self.rect_item.scene():
                self.rect_item.scene().removeItem(self.rect_item)
            self.rect_item = QGraphicsRectItem(final_rect)
            self.rect_item.setPen(QPen(QColor(0,255,0),2))
            scene.addItem(self.rect_item)
            if self.temp_rect_item and self.temp_rect_item.scene():
                self.temp_rect_item.scene().removeItem(self.temp_rect_item)
            self.temp_rect_item = None
            self.start_pos = None
            return True
        return False

    # 高亮表格
    def highlight_table_row(self, row_index):
        # 清除已有选择
        self.tableView.clearSelection()
        # 设置选择模式为行选择
        self.tableView.selectRow(row_index)
        # 切换到第二个 tab（结果展示视图）
        self.tabWidget.setCurrentIndex(1)
        # 可以添加：居中滚动到该行
        index = self.model.index(row_index, 0)
        self.tableView.scrollTo(index, QtWidgets.QAbstractItemView.PositionAtCenter)

    # 高亮矩形框
    def highlight_rect_item(self, index):
        for i, item in enumerate(self.rect_items):
            if i == index:
                item.border.setPen(QPen(Qt.blue, 5))
                # 可选：居中视图到这个矩形
                center = item.pos()
                self.rectView.centerOn(center)
            else:
                item.border.setPen(QPen(Qt.red, 2))

    # 识别框是否被选中
    def on_selection_changed(self):
        selected_items = self.rect_scene.selectedItems()
        for item in self.rect_scene.items():
            if isinstance(item, ResizableRotatableRectItem):
                previously_active = item.activate_flag
                currently_active = item in selected_items

                item.activate_flag = currently_active

                if currently_active and not previously_active:
                    # 记录选中时的角度
                    item.last_rotation = -item.rotation()
                    # print(f"✅ 选中 Item at {item.pos()} 当前角度 = {item.last_rotation:.2f}")
                elif not currently_active and previously_active:
                    if item.rotation() == 0.0:
                        self.current_rotation = 0.0
                    else:
                        self.current_rotation = -item.rotation()
                    row = item.index
                    txt = QtGui.QStandardItem(str(self.current_rotation))
                    self.model.setItem(row,2,txt)

                    red_handle = item.handles[0]  # index 0 是左上角，即红色控制点
                    red_handle_pos = red_handle.mapToScene(red_handle.boundingRect().center())
                    x = red_handle_pos.x()
                    y = red_handle_pos.y()
                    x, y = edr2.trans_coord(x / self.tran_x, y / self.tran_y, self.attr0)
                    x -= self.min_x / self.tran_x
                    y = self.max_y / self.tran_y - y
                    pos_item = QtGui.QStandardItem(f"{x:.1f},{y:.1f}")
                    self.model.setItem(row,1,pos_item)

    # 表格删除功能
    def handle_table_click(self, index):
        row = index.row()
        col = index.column()
        if col == 4:
            # 1.删除scene中的图形项
            if row < len(self.rect_items):
                item = self.rect_items.pop(row)
                if item.scene():
                    item.scene().removeItem(item)
            # 2.删除起始点中的对应数据
            # if self.ell_items[row]:
            #     ell = self.ell_items.pop(row)
            #     ell.scene().removeItem(ell)
            # 3. 删除模型中的这一行
            self.model.removeRow(row)

            for i, item in enumerate(self.rect_items):
                item.index = i
        else:
            self.highlight_rect_item(row)

    def clear_all(self):
        if self.is_clear:
            # 清空视图
            self.svg_scene.clear()  # 清空 SVG 视图的内容
            self.rect_scene.clear()  # 清空框选矩形视图的内容
            # 清空表格内容
            self.table.clearContents()
            self.table.setRowCount(0)
            self.repalce_table.clearContents()
            self.repalce_table.setRowCount(0)
            # 清空模型数据
            self.tableView.setModel(None)  # 断开表格与模型的绑定
            if self.model:
                self.model.clear()  # 安全清空模型内容
            self.tableView.setModel(self.model)  # 重新绑定模型（如果还需要继续使用）
            self.init_table()  # 表格初始化

            self.textEdit.clear() # 清空多行文本
            # 设置按钮状态
            self.rect_button.setEnabled(False)
            self.det_button.setEnabled(False)
            self.add_button.setEnabled(False)
            self.export_Button.setEnabled(False)
            self.replace_button.setEnabled(False)

            # self.filepath = None  # pdf文件路径
            # self.filepath_svg = None  # svg文件路径
            self.is_clear = False  # 清空标志
            self.rect_item = None  # 识别区域矩形框
            self.flag_add = False  # 添加识别框标志位
            self.flag_rect = False  # 绘制识别区域矩形框标志位
            self.temp_rect_item = None  # 框选识别区域的临时矩形框
            self.start_pos = None  # 框选识别区域的起点
            self.rect_items = []  # 识别框列表
            self.current_rotation = None  # 修改后的识别框角度

            self.max_length.setText('500')
            self.str_length.setText('10')


            # pdf中上下表格信息提取的框
            self.rectItem0 = []
            self.rectItem1 = []
            self.rectItem2 = []
            self.rectItem3 = []
            self.rectItem4 = []
            self.rectItem5 = []
            self.rectItem6 = []
            self.rectItem7 = []

            # 坐标变换系数
            self.tran_x = None
            self.tran_y = None

            self.min_x = None
            self.max_x = None
            self.min_x = None
            self.max_x = None
            self.length = None
            self.width = None

    def load_file(self):
        self.filepath, _ = QFileDialog.getOpenFileName(
            self,
            "打开PDF文件",
            "",
            "PDF Files (*.pdf)"
        )

        if not self.filepath:
            return
        else:
            self.clear_all()
            self.filename = Path(self.filepath).stem
            import subprocess
            import os
            output_svg_path = os.path.join(os.getcwd(), "output.svg")
            self.filepath_svg = output_svg_path

            # 调用inkscape 转换
            inkscape_path = edr2.resource_path("inkscape/bin/inkscape.exe")
            result = subprocess.run([
                # r"E:\Code_Tool\inkscape\bin\inkscape.exe",
                # r"inkscape\bin\inkscape.exe",
                inkscape_path,
                self.filepath,
                "--export-type=svg",
                f"--export-filename={output_svg_path}"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                # print("Inkscape 转换失败:", result.stderr)
                QMessageBox.critical(
                    self, "转换失败",
                    f"Inkscape 报错：\n\n{result.stderr or result.stdout}"
                )
                return
            # 取出transform
            ps, ats = svg2paths(self.filepath_svg)
            self.attr0 = ats[1]
            self.svg_scene.clear()
            # 创建 SVG 图元
            svg_item = QGraphicsSvgItem(self.filepath_svg)
            svg_item.setFlags(svg_item.flags() | svg_item.ItemClipsToShape)
            svg_item.setCacheMode(svg_item.NoCache)
            svg_item.setZValue(0)  # 可选：图层深度
            # 添加到画布
            self.svg_scene.addItem(svg_item)
            # 可选：设置 scene 大小适配 SVG 大小
            renderer = QSvgRenderer(self.filepath_svg)
            self.svg_scene.setSceneRect(renderer.viewBoxF())
            self.svgView.fitInView(self.svg_scene.sceneRect(), Qt.KeepAspectRatio)
            # 坐标变换
            # svg画布尺寸
            import xml.etree.ElementTree as ET
            tree = ET.parse(output_svg_path)
            root = tree.getroot()
            # 获取命名空间前缀（如果有）
            ns = {'svg': 'http://www.w3.org/2000/svg'}

            # 获取 width 和 height 属性
            def parse_svg_size(value):
                if value is None:
                    return None
                return float(''.join(c for c in value if c.isdigit() or c == '.'))

            width_str = root.get('width')
            height_str = root.get('height')
            viewBox = root.get('viewBox')
            if viewBox:
                # 使用 viewBox 优先，格式如 "0 0 1190 1684"
                vb_values = list(map(float, viewBox.strip().split()))
                svg_width, svg_height = vb_values[2], vb_values[3]
            else:
                svg_width = parse_svg_size(width_str)
                svg_height = parse_svg_size(height_str)

            # PDF 页面尺寸
            doc = fitz.open(self.filepath)
            page = doc[0]
            pdf_width, pdf_height = page.rect.width, page.rect.height
            # print(f'pdf尺寸：宽{pdf_width},高{pdf_height}')
            # print(f'svg尺寸：宽{svg_width},高{svg_height}')

            # 坐标变换
            scale_x = svg_width / pdf_width
            scale_y = svg_height / pdf_height

            self.tran_x = scale_x
            self.tran_y = scale_y
            # 应用到矩形上
            self.rectItem0 = ResizableRectItem(
                QRectF(685 * scale_x, 75 * scale_y, 175 * scale_x, 20 * scale_y),
                label_text='材料规格'
            )
            self.rectItem1 = ResizableRectItem(
                QRectF(810 * scale_x, 100 * scale_y, 50 * scale_x, 25 * scale_y),
                label_text='材质'
            )
            self.rectItem2 = ResizableRectItem(
                QRectF(980 * scale_x, 40 * scale_y, 170 * scale_x, 30 * scale_y),
                label_text='切割指令号'
            )
            self.rectItem3 = ResizableRectItem(
                QRectF(978 * scale_x, 126 * scale_y, 40 * scale_x, 22 * scale_y),
                label_text='比例'
            )
            # 下方表格区域
            self.rectItem4 = ResizableRectItem(
                QRectF(78 * scale_x, 554 * scale_y, 222 * scale_x, 246 * scale_y),
                label_text='第一列'
            )
            self.rectItem5 = ResizableRectItem(
                QRectF(348 * scale_x, 554 * scale_y, 222 * scale_x, 246 * scale_y),
                label_text='第二列'
            )
            self.rectItem6 = ResizableRectItem(
                QRectF(618 * scale_x, 554 * scale_y, 222 * scale_x, 246 * scale_y),
                label_text='第三列'
            )
            self.rectItem7 = ResizableRectItem(
                QRectF(888 * scale_x, 554 * scale_y, 222 * scale_x, 246 * scale_y),
                label_text='第四列'
            )

            self.rectItem0.setPen(QPen(Qt.red, 2))
            self.rectItem1.setPen(QPen(Qt.red, 2))
            self.rectItem2.setPen(QPen(Qt.red, 2))
            self.rectItem3.setPen(QPen(Qt.red, 2))
            self.rectItem4.setPen(QPen(Qt.red, 2))
            self.rectItem5.setPen(QPen(Qt.red, 2))
            self.rectItem6.setPen(QPen(Qt.red, 2))
            self.rectItem7.setPen(QPen(Qt.red, 2))
            self.svg_scene.addItem(self.rectItem0)
            self.svg_scene.addItem(self.rectItem1)
            self.svg_scene.addItem(self.rectItem2)
            self.svg_scene.addItem(self.rectItem3)
            self.svg_scene.addItem(self.rectItem4)
            self.svg_scene.addItem(self.rectItem5)
            self.svg_scene.addItem(self.rectItem6)
            self.svg_scene.addItem(self.rectItem7)
            self.tabWidget.setCurrentIndex(0)
            self.tabWidget_2.setCurrentIndex(0)
            self.rect_button.setEnabled(True)

            self.is_clear = True

    def table_ocr(self):
        above_data = table.ocr_key_regions(self, self.filepath, page_number=0, )
        data = table.extract_pdf_data(self, self.filepath, page_number=0, zoom=10)
        parsed_data = [row for row in data if "序号" in row]  # 过滤掉未能解析的行（即只保留包含“序号”的数据）
        # 将数据导入table
        self.table.setRowCount(len(parsed_data))
        for row_idx, row in enumerate(parsed_data):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row["序号"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(row["零件编码"]))
            self.table.setItem(row_idx, 2, QTableWidgetItem(row["尺寸"]))

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)

        value = [line.split(":", 1)[1].strip() for line in above_data]
        self.textEdit.setPlainText(f"材料规格：{value[0]}\n材质：{value[1]}\n切割指令号：{value[2]}\n比例：{value[3]}")

    # 绘制ocr识别区域
    def enable_rect_selection(self):
        self.tabWidget.setCurrentIndex(0)
        self.det_button.setEnabled(True)
        self.flag_rect = True

    def svg_ocr(self):
        try:
            max_length_val = float(self.max_length.text())
            str_length_val = float(self.str_length.text())

        except ValueError:
            QtWidgets.QMessageBox.warning(self, "输入错误", "请输入有效的数值")
            return

        # 禁用按钮避免重复点击
        self.det_button.setEnabled(False)

        region = (
            self.canvas_xmin, self.canvas_xmax,
            self.canvas_ymin, self.canvas_ymax
        )

        # 创建线程
        self.ocr_thread = SvgOcrThread(
            file_path=self.filepath_svg,
            max_length_val=max_length_val,
            str_length_val=str_length_val,
            region=region
        )

        self.ocr_thread.progress.connect(self.update_progress)
        self.ocr_thread.finished.connect(self.handle_ocr_result)
        self.ocr_thread.error.connect(lambda msg: QtWidgets.QMessageBox.critical(self, "出错", msg))
        self.ocr_thread.start()

        above_data = table.ocr_key_regions(self, self.filepath, page_number=0, )
        data = table.extract_pdf_data(self, self.filepath, page_number=0, zoom=10)
        parsed_data = [row for row in data if "序号" in row]  # 过滤掉未能解析的行（即只保留包含“序号”的数据）
        # 将数据导入table
        self.table.setRowCount(len(parsed_data))
        for row_idx, row in enumerate(parsed_data):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(row["序号"])))
            self.table.setItem(row_idx, 1, QTableWidgetItem(row["零件编码"]))
            self.table.setItem(row_idx, 2, QTableWidgetItem(row["尺寸"]))

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)

        value = [line.split(":", 1)[1].strip() for line in above_data]
        self.textEdit.setPlainText(f"材料规格:{value[0]}\n材质:{value[1]}\n切割指令号:{value[2]}\n比例:{value[3]}")

    def handle_ocr_result(self,data):
        results, self.filtered_paths,self.min_x, self.max_x, self.min_y, self.max_y = data
        self.length = self.max_x/ self.tran_x - self.min_x/ self.tran_x
        self.width = self.max_y/ self.tran_y - self.min_y/ self.tran_y
        # print(
        #     f'变换的边框数据{self.min_x / self.tran_x},{self.max_x / self.tran_x},{self.min_y / self.tran_y},{self.max_y / self.tran_y}')

        self.model.removeRows(0, self.model.rowCount())
        # 将数据插入表格
        results_filter = []
        for result in results:
            text = result.text[0] if isinstance(result.text, list) and result.text else ""
            if text:
                if re.fullmatch(r'[a-zA-Z]',text):
                    continue
                results_filter.append(result)
                str_item = QtGui.QStandardItem(text)
                score = sum(float(i) for i in result.score) / len(result.score)
                score_item = QtGui.QStandardItem(f"{score:.2f}")
                score_item.setFlags(score_item.flags() & ~Qt.ItemIsEditable)
                if score < 0.95:
                    score_item.setBackground(QBrush(QColor(255, 200, 200)))
                x,y = result.startPoint
                x, y = edr2.trans_coord(x/self.tran_x, y/self.tran_y, self.attr0)
                x -= self.min_x/ self.tran_x
                y = self.max_y/ self.tran_y - y
                pos_item = QtGui.QStandardItem(f"{x:.1f},{y:.1f}")
                # print(result.angle)
                # print(type(result.angle))
                angle = QtGui.QStandardItem(f'{result.angle:.2f}')
                op_item = QtGui.QStandardItem("删除")
                op_item.setFlags(op_item.flags() & ~Qt.ItemIsEditable)
                self.model.appendRow([str_item, pos_item,angle, score_item, op_item])
        # 视图绘制
        self.rect_scene.clear()
        for path, attr in self.filtered_paths:  # 注意 filtered_paths 要从线程传回或缓存
            stroke_width = self.get_stroke_width(attr)
            subpaths = path.continuous_subpaths() if hasattr(path, 'continuous_subpaths') else [path]
            qt_path = QPainterPath()
            for subpath in subpaths:
                for i, seg in enumerate(subpath):
                    if i == 0:
                        qt_path.moveTo(seg.start.real, seg.start.imag)
                    qt_path.lineTo(seg.end.real, seg.end.imag)
            item = QGraphicsPathItem(qt_path)
            item.setPen(QPen(Qt.black, stroke_width))
            self.rect_scene.addItem(item)
        self.rectView.resetTransform() #缩放图形视图
        self.rectView.scale(0.1, 0.1)


        # 绘制可交互的矩形框
        for index, result in enumerate(results_filter):
            # x,y = result.startPoint
            # radius = 5
            # ellipse = QGraphicsEllipseItem(QRectF(x - radius,y - radius,radius*2,radius*2))
            # ellipse.setBrush(QBrush(QColor('red')))
            # ellipse.setPen(QColor('black'))  # 边框颜色
            # self.rect_scene.addItem(ellipse)
            # self.ell_items.append(ellipse)
            w = result.width
            h = result.height
            angle = result.angle
            text = result.text[0] if isinstance(result.text, list) and result.text else ""
            if re.fullmatch(r'\d', text):
                rectItem = ResizableRotatableRectItem(QRectF(-h / 2, -w / 2, h, w), angle, index,
                                                      self.highlight_table_row)
            else:
                rectItem = ResizableRotatableRectItem(QRectF(-w/2,-h/2,w,h),angle,index,self.highlight_table_row)
            # rectItem.setFlag(QGraphicsItem.ItemIsMovable,False)
            cx,cy = result.centerpoint
            rectItem.setPos(cx,cy) # 以矩形框中心点为参考点，设置矩形框的场景坐标
            self.rect_items.append(rectItem)
            self.rect_scene.addItem(rectItem)

        self.add_button.setEnabled(True)
        self.replace_button.setEnabled(True)
        self.export_Button.setEnabled(True)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(1)

    def get_stroke_width(self,attr, default=1.0):
        style_str = attr.get('style', '')
        for item in style_str.split(';'):
            if 'stroke-width' in item:
                try:
                    return float(item.split(':')[1])
                except:
                    return default
        return default

    # 添加识别框
    def add_rect(self):
        # 获取当前视图的中心点坐标
        center_in_view = self.rectView.viewport().rect().center()
        center_in_scene = self.rectView.mapToScene(center_in_view)
        x = center_in_scene.x()
        y = center_in_scene.y()
        # print(x,y)
        w,h = 120,80
        rectItem = ResizableRotatableRectItem(QRectF(-w / 2, -h / 2, w, h), angle = 0.0, index = len(self.rect_items), callback=self.highlight_table_row)
        rectItem.setPos(x,y)
        drop_o = QtGui.QStandardItem("删除")
        drop_o.setFlags(drop_o.flags() & ~Qt.ItemIsEditable)  # 取消可编辑
        self.model.setItem(rectItem.index,4,drop_o)
        self.model.setItem(rectItem.index,3,QtGui.QStandardItem("None"))
        self.rect_items.append(rectItem)
        self.rect_scene.addItem(rectItem)

        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(1)

    # 替换数据
    def replace_data(self):
        if self.table.rowCount() == 0:
            QMessageBox.critical(self, '警告', '请先点击表格识别')
            return
        self.repalce_table.setRowCount(self.model.rowCount())
        # 将table上的数据制作成字典
        mapping = {}
        for row in range(self.table.rowCount()):
            item_num = self.table.item(row, 0)
            item_code = self.table.item(row, 1)
            if item_num and item_code:
                mapping[item_num.text()] = item_code.text()
        data = []
        for row in range(self.model.rowCount()):
            txt = self.model.data(self.model.index(row, 0))
            pos = self.model.data(self.model.index(row, 1))
            angle = self.model.data(self.model.index(row,2))
            Angle = -float(angle)
            self.repalce_table.setItem(row, 1, QTableWidgetItem(pos if pos else ''))
            # self.repalce_table.setItem(row, 2, QTableWidgetItem(angle if angle else ''))
            item = QTableWidgetItem(str(Angle)) if Angle else QTableWidgetItem('0.0')
            self.repalce_table.setItem(row, 2, item)
            self.repalce_table.setItem(row, 3, QTableWidgetItem('40'))
            # 替换文本：用零件编码代替序号
            if txt:
                replace_txt = mapping.get(txt.strip())
                if replace_txt:
                    self.repalce_table.setItem(row, 0, QTableWidgetItem(replace_txt))
                else:
                    self.repalce_table.setItem(row, 0, QTableWidgetItem(txt))
            else:
                self.repalce_table.setItem(row, 0, QTableWidgetItem(""))
        self.tabWidget_2.setCurrentIndex(2)
        self.export_Button.setEnabled(True)
        # self.gen_button.setEnabled(True)

    def update_progress(self, val: float):
        self.progressBar.setValue(int(val))  # 进度条的值必须是整数
        self.progressBar.setFormat(f"{val:.1f}%")  # 显示一位小数的百分比文本

    def create_gen(self,*, suffix: str = ".gen"):
        config_path = edr2.resource_path1("nn.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        name = self.filename
        out_dir = self.exe_dir()
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{name}{suffix}"

        Info = self.get_line()  # 假设是 list[str]，每行一条
        # print(Info[0])
        # print(Info[1])
        # print(Info[2])
        # print(Info[3])

        # 规格解析：例如 "规格: Q235*4000*2000"
        guige_match = re.search(r":\s*(.+)", Info[0])
        # print(f'guige_match={guige_match}')
        if guige_match:
            guige_list = guige_match.group(1).split("*")
            THICKNESS = guige_list[0] if len(guige_list) > 0 else ""
            LENGTH = guige_list[1] if len(guige_list) > 1 else ""
            WIDTH = guige_list[2] if len(guige_list) > 2 else ""
        else:
            THICKNESS = LENGTH = WIDTH = ""

        # print(f'THICKNESS={THICKNESS}')
        # print(f'LENGTH={LENGTH}')
        # print(f'WIDTH={WIDTH}')

        # 材质解析：例如 "材质: Q235"
        caizhi_match = re.search(r":\s*(.+)", Info[1])
        caizhi = caizhi_match.group(1) if caizhi_match else ""
        # print(f'caizhi={caizhi}')
        # print(f'caizhi_match={caizhi_match}')

        # 切割指令：例如 "切割: PLASMA"
        qiegezhiling_match = re.search(r":\s*(.+)", Info[2])
        qiegezhiling = qiegezhiling_match.group(1) if qiegezhiling_match else ""
        # print(f'qiegezhiling_match={qiegezhiling_match}')
        # print(f'qiegezhiling={qiegezhiling}')

        # 比例：例如 "比例: 1:1"
        pattern = r"[：:]\s*(\d+)\s*[:：]\s*(\d+)"
        bili_match = re.search(pattern, Info[3])
        if bili_match:
            num = int(bili_match.group(1))  # 1
            den = int(bili_match.group(2))  # 16
            b = den
        # print(f'bili_match={bili_match}')
        # print(f'b={b}')

        with open(path, 'w', encoding='utf-8') as f:
            f.write(
                f"{config['start0']}\n"
                f"{config['start1']}\n"
                f"{config['start2']}\n"
                f"{config['start3']}\n"
                f"{config['start4']}{qiegezhiling}\n"
                f"{config['start5']}\n"
                f"{config['start6']}{LENGTH}\n"
                f"{config['start7']}{WIDTH}\n"
                f"{config['start8']}{THICKNESS}\n"
                f"{config['start9']}\n"
                f"{config['start10']}\n"
                f"{config['start11']}\n"
                f"{config['start12']}\n"
                f"{config['start13']}\n"
                f"{config['start14']}\n"
                f"{config['start15']}\n"
                f"{config['start16']}\n"
                f"{config['start17']}\n"
                f"{config['start18']}\n"
                f"{config['start19']}\n"
                f"{config['start20']}\n"
                f"{config['start21']}\n"
                f"{config['start22']}\n"
                f"{config['start23']}\n"
                f"{config['start24']}\n"
                f"{config['start25']}\n"
                f"{config['start26']}\n"
                f"{config['start27']}\n"
                f"{config['start28']}\n"
                f"{config['start29']}\n"
                f"{config['start30']}\n"
                f"{config['start31']}\n"
                f"{config['start32']}\n"
                f"{config['start33']}\n"
                f"{config['start34']}\n"
                f"{config['start35']}\n"
                f"{config['start36']}\n"
                f"{config['start37']}{date.today()}\n"
                f"{config['start38']}\n"
                f"{config['start39']}\n"
                f"\n"
            )

            for row in range(self.repalce_table.rowCount()):
                textitem = self.repalce_table.item(row, 0)
                text = textitem.text().strip() if textitem else ""

                positem = self.repalce_table.item(row, 1)
                pos = positem.text().strip() if positem else ""
                poses = pos.split(',') if ',' in pos else ["", ""]
                u = poses[0].strip() if len(poses) > 0 else ""
                v = poses[1].strip() if len(poses) > 1 else ""
                U = float(u)*self.length/float(LENGTH)
                V = float(v)*self.width/float(WIDTH)
                angleitem = self.repalce_table.item(row, 2)
                angle = angleitem.text().strip() if angleitem else ""
                heightitem = self.repalce_table.item(row, 3)
                height = heightitem.text().strip() if textitem else ""

                # f.write(f"LABELTEXT_DATA\nTEXT_POSITION_U={u}\nTEXT_POSITION_V={v}\nTEXT_ANGLE={angle}\n"
                #         f"TEXT_HEIGHT={height}\nTEXT={text}\nEND_OF_LABELTEXT_DATA\n")
                f.write(
                    f"{config['value0']}\n"
                    f"{config['value7']}\n"
                    f"{config['value1']}{U}\n"
                    f"{config['value2']}{V}\n"
                    f"{config['value3']}{angle}\n"
                    f"{config['value4']}{height}\n"
                    f"{config['value5']}{text}\n"
                    f"{config['value6']}\n"
                )

        # ✅ 写入完成后弹出提示框
        QMessageBox.information(self, "完成", f"文件已成功生成：\n{path}")

        return path

    def get_line(self):
        lines = self.textEdit.toPlainText().splitlines()
        try:
            return lines
        except IndexError:
            return ""

    from pathlib import Path

    def exe_dir(self) -> Path:
        """返回当前程序（脚本 / PyInstaller-exe）的所在目录 Path"""
        if getattr(sys, "frozen", False):  # PyInstaller 打包后
            return Path(sys.executable).parent  # xxx.exe 所在目录
        else:  # 源码运行
            return Path(__file__).resolve().parent  # 当前 .py 文件目录



if __name__ == '__main__':
    import sys
    # import multiprocessing
    # multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())