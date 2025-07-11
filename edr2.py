import re
import io
import cv2
import math
import numpy as np
from PIL import Image
from rapidocr import RapidOCR
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Rectangle,Polygon
from svgpathtools import svg2paths, parse_path, Line,Path
import debugTool
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PathInfo:
    def __init__(self,path=None,rect=None,centralPoint=None,width=None,height=None):
        self.path = path
        self.rect = rect
        self.centralPoint = centralPoint
        self.width = width
        self.height = height

class StringInfo:
    def __init__(self,num = None,path = None,text = None,score = None,startPoint = None,width = None,height= None,angle = None,centerpoint=None):
        self.text = text
        self.path = path
        self.score = score
        self.num = num
        self.startPoint = startPoint
        self.width = width
        self.height = height
        self.angle = angle
        self.centerpoint = centerpoint

def get_transform_coordinate(region, attr):
    xmin, xmax, ymin, ymax = region
    transform_str = attr.get('transform', '')

    # matrix 变换
    matrix_match = re.search(
        r'matrix\(\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+)\s*\)',
        transform_str)

    if matrix_match:
        a, b, c, d, e, f = map(float, matrix_match.groups())
        # 左下角 (xmin, ymin)，右上角 (xmax, ymax)
        x0 = a * xmin + c * ymin + e
        x1 = a * xmax + c * ymax + e
        y0 = b * xmin + d * ymin + f
        y1 = b * xmax + d * ymax + f
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    # scale 变换
    scale_match = re.search(r'scale\(\s*([\d\.\-eE]+)(?:\s*,\s*([\d\.\-eE]+))?\s*\)', transform_str)
    if scale_match:
        sx = float(scale_match.group(1))
        sy = float(scale_match.group(2)) if scale_match.group(2) else sx
        return (xmin * sx, xmax * sx, ymin * sy, ymax * sy)

    # 无变换
    return (xmin, xmax, ymin, ymax)

def is_path_in_region(path, attr, xmin, xmax, ymin, ymax):
    # 获取路径的原始边界框
    path_xmin1, path_xmax1, path_ymin1, path_ymax1 = path.bbox()

    # 进行变换后的 bbox 获取
    path_xmin, path_xmax, path_ymin, path_ymax = get_transform_coordinate(
        (path_xmin1, path_xmax1, path_ymin1, path_ymax1),
        attr
    )

    # 判断是否与选区有交集
    return not (path_xmax < xmin or path_xmin > xmax or
                path_ymax < ymin or path_ymin > ymax)

def get_stroke_width(attr):
    style = attr.get('style', '')
    # 从style字符串里用正则找stroke-width，单位可能有px或无单位
    match = re.search(r'stroke-width\s*:\s*([\d\.]+)', style)
    if match:
        return float(match.group(1))
    else:
        # 默认线宽为1，或你自己定义的默认值
        return 1.0

def group_paths_by_stroke_width(filtered_paths):
    groups = defaultdict(list)
    for path, attr in filtered_paths:
        sw = get_stroke_width(attr)
        groups[sw].append((path, attr))
    return groups

def filter_remain_continuous_paths(filtered_paths):
    """
    删除所有使用虚线绘制的路径（即 style 中 stroke-dasharray 不是 none 的路径）。

    :param filtered_paths: [(path, attr), ...] 列表
    :return: 不包含虚线路径的列表
    """
    conti_paths = []

    for path, attributes in filtered_paths:
        style = attributes.get('style', '')
        match = re.search(r'stroke-dasharray\s*:\s*([^;]+)', style)
        if match:
            dash_value = match.group(1).strip()
            if dash_value.lower() == 'none':
                conti_paths.append((path, attributes))

    return conti_paths

#第一次处理
def first_pretreatment(paths,min_length = 500):
    # 删除过长直线和曲线
    result0 = filter_short_paths(paths,min_length)

    # ID分组
    result1 = group_filtered_paths_by_consecutive_id(result0)

    # path画框判重叠
    rect_path_group = []
    for pathlist in result1:
        rect_path_list = []
        for element in pathlist:
            rect_path_list.append(plot_box_path(element))
        rect_path_group.append(rect_path_list.copy())

    path_group = []
    for e in rect_path_group:
        path_list = []
        i = 1
        while i < len(e):
            if is_overlap(e[i - 1].rect, e[i].rect):
                e[i - 1].path.extend(e[i].path)
                del e[i]
            else:
                i += 1
        j = 0
        while j < len(e):
            path_list.append(e[j].path)
            j += 1
        path_group.append(path_list)

    return  path_group

class StrAndAngle:
    def __init__(self,strPath=None, angle = None):
        self.strPath = strPath
        self.angle = angle

class SecondResult:
    def __init__(self,path = None,centerpoint = None):
        self.paths = path
        self.centerpoints = centerpoint

# 第二次处理
def second_pretreatment(first_paths,string_thre = 10):
    string_straight_group = []  # 水平或竖直字符串
    string_incline_group = []  # 倾斜字符串
    string_single_group = []  # 单一字符
    for element in first_paths:
        string_straight_list = []  # 一组规整字符串
        string_incline_list = []  # 一组倾斜字符串
        string_single_list = []  # 一组单一字符
        temp_pathlist_straight = []
        temp_pathlist_incline = []
        tstring_single_list = []
        left_string_list = []
        temp_centerpoint = []
        temp_centerpoint1 = []
        temp_centerpoint2 = []

        # 筛选规整字符串
        i = 1
        while i < len(element):
            result0 = plot_box_pathlist(element[i])
            result1 = plot_box_pathlist(element[i-1])
            w = result0.width
            h = result0.height
            (cx,cy) = result0.centralPoint
            (cx1,cy1) = result1.centralPoint
            path = result0.path
            path1 = result1.path

            if w < h:    # 即水平边为短边
                thre_x,thre_y = (w+h),0
            else:  # 即竖直边为短边
                thre_x, thre_y = 0, (w+h)

            if abs(cx - cx1) <= thre_x and abs(cy - cy1) <= thre_y: # 判断是否是水平或竖直字符串
                temp_pathlist_straight.extend(path1)
                temp_centerpoint.append((cx1, cy1))
                if i == len(element)-1:
                    temp_pathlist_straight.extend(path)
                    temp_centerpoint.append((cx,cy))
                    string_straight_list.append(SecondResult(temp_pathlist_straight.copy(),temp_centerpoint.copy()))
            else:
                if temp_pathlist_straight:
                    temp_pathlist_straight.extend(path1)
                    temp_centerpoint.append((cx1, cy1))
                    string_straight_list.append(SecondResult(temp_pathlist_straight.copy(),temp_centerpoint.copy()))
                    temp_pathlist_straight.clear()
                    temp_centerpoint.clear()
                else:
                    left_string_list.append(element[i-1])
                    if i == len(element)-1:
                        left_string_list.append(element[i])
            i += 1

        # 筛选倾斜字符串
        i = 1
        while i < len(left_string_list):
            result2 = plot_box_pathlist(left_string_list[i])
            result3 = plot_box_pathlist(left_string_list[i - 1])
            (cx,cy) = result2.centralPoint
            (cx1,cy1) = result3.centralPoint
            w = result2.width
            h = result2.height
            path = result2.path
            path1 = result3.path


            if math.sqrt((cx - cx1) ** 2 + (cy - cy1) ** 2) <= (w+h):
                temp_pathlist_incline.extend(path1)
                temp_centerpoint1.append((cx1, cy1))
                if i == len(left_string_list)-1:
                    temp_pathlist_incline.extend(path)
                    temp_centerpoint1.append((cx, cy))
                    string_incline_list.append(SecondResult(temp_pathlist_incline.copy(),temp_centerpoint1.copy()))
                i += 1
            else:
                if temp_pathlist_incline:
                    temp_pathlist_incline.extend(path1)
                    temp_centerpoint1.append((cx1, cy1))
                    string_incline_list.append(SecondResult(temp_pathlist_incline.copy(),temp_centerpoint1.copy()))
                    temp_pathlist_incline.clear()
                    temp_centerpoint1.clear()
                else:
                    tstring_single_list.extend(path1)
                    temp_centerpoint2.append((cx1,cy1))
                    string_single_list.append(SecondResult(tstring_single_list.copy(),temp_centerpoint2.copy()))
                    tstring_single_list.clear()
                    temp_centerpoint2.clear()
                    if i == len(left_string_list)-1:
                        tstring_single_list.extend(path)
                        temp_centerpoint2.append((cx, cy))
                        string_single_list.append(SecondResult(tstring_single_list.copy(),temp_centerpoint2.copy()))
                i += 1

        string_straight_group.extend(second_filter(string_straight_list,string_thre))
        string_incline_group.extend(second_filter(string_incline_list,string_thre))
        string_single_group.extend(second_filter(string_single_list,string_thre))

    return string_straight_group, string_incline_group, string_single_group

# 生成图片，OCR识别
def create_images_ocr(groups, resize=(5, 5), save_path='OutputImage/', dpi=300, padding=300, flip_vertical=True,stop_flag = None, progress_callback=None, engine = None):
# def create_images_ocr(groups, resize=(3, 3), save_path='OutputImage/', dpi=300, padding=50, flip_vertical=True,stop_flag = None, progress_callback=None, engine = None):
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    All_len = 0
    if groups:
        All_len = len(groups)
    # print(f'识别长度为：{All_len}')

    string_info = []

    for idx, e in enumerate(groups):
        if stop_flag is not None and not stop_flag():
            break

        # 识别进度百分比更新
        if progress_callback is not None:
            progress_callback(1)
        fig, ax = plt.subplots(figsize=resize)

        all_x = []
        all_y = []

        pathlist = e.strPath
        for path, attr in pathlist:
            for segment in path:
                points = [segment.start, segment.end]
                all_x.extend(p.real for p in points)
                all_y.extend(p.imag for p in points)

        if not all_x or not all_y:
            continue

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        for path, attr in e.strPath:
            for segment in path:
                points = [segment.start, segment.end]
                xs = [p.real for p in points]
                ys = [p.imag for p in points]
                ax.plot(xs, ys, color='black', linewidth=2)

        ax.set_aspect('equal')
        ax.invert_yaxis()  # 加上这一行！！
        ax.axis('off')

        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)

        plt.close(fig)

        # 绘制完fig, ax以后
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, pad_inches=0)
        buf.seek(0)

        # 从内存中读出为PIL.Image
        img = Image.open(buf).convert("RGB")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)


        angle = e.angle
        w, h = img.size
        center = (w // 2, h // 2)
        # 绕中心旋转，使用高质量 BICUBIC
        rotated_img = img.rotate(angle, resample=Image.BICUBIC, center=center)
        # debugTool.show_inter_image(rotated_img)
        result = engine(rotated_img)
        a = float(result.scores[0]) if result.scores and result.scores[0] is not None else 0.0
        if a > 0.8 and result.txts:
            text = [re.sub(r'[^A-Z0-9]', '', line) for line in result.txts]
            startPoint,centerPoint,width,height ,Angle= get_rect_data(e)
            temp = StringInfo(text=text,path=pathlist, score=result.scores,num=idx+1,startPoint = startPoint,width = width,height = height,angle=Angle,centerpoint=centerPoint)
            string_info.append(temp)
            # print(text)

        # print(f'第{idx}次')
        # if idx > 10:
        #     break
    return string_info

# 删除长度超过阈值的直线和曲线
def filter_short_paths(filtered_paths, min_length):
    """
    删除出总长度大于 min_length 的路径（可包含直线或曲线段）。

    :param filtered_paths: [(path, attr), ...]
    :param min_length: 长度阈值，默认 100
    :return: 满足条件的 [(path, attr), ...]
    """
    short_paths = []

    for path, attr in filtered_paths:
        total_length = sum(seg.length() for seg in path)
        if total_length < min_length:
            short_paths.append((path, attr))

    return short_paths

def extract_id_number(attr):
    """
    从路径属性中提取 ID 数字，例如 'path123' -> 123
    """
    path_id = attr.get('id', '')
    match = re.search(r'(\d+)', path_id)
    return int(match.group(1)) if match else None

# ID分组
def group_filtered_paths_by_consecutive_id(filtered_paths):
    """
    按照路径 ID 的数字部分连续进行分组。

    :param filtered_paths: [(path, attr), ...]
    :return: 分组后的路径列表 [[(path, attr), ...], [(path, attr), ...], ...]
    """
    # 提取 ID 编号
    numbered_paths = []
    for path, attr in filtered_paths:
        id_num = extract_id_number(attr)
        if id_num is not None:
            numbered_paths.append((id_num, path, attr))

    # 按 ID 数值排序
    numbered_paths.sort(key=lambda x: x[0])

    # 分组（按连续编号）
    groups = []
    current_group = [numbered_paths[0][1:]]  # 只保留 (path, attr)

    for (prev_id, _, _), (curr_id, path, attr) in zip(numbered_paths, numbered_paths[1:]):
        if curr_id == prev_id + 1:
            current_group.append((path, attr))
        else:
            groups.append(current_group)
            current_group = [(path, attr)]

    groups.append(current_group)

    # 对每组进行判断：若全是直线，则删除

    return groups

def plot_box_path(pathl):
    path_list = []
    path_list.append(pathl)
    path,attr = pathl
    all_x,all_y = [],[]

    for segment in path:
        all_x.extend([segment.start.real, segment.end.real])
        all_y.extend([segment.start.imag, segment.end.imag])

    # 为路径绘制其外接矩形框（最小宽高保护）
    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        width = xmax - xmin
        height = ymax - ymin

        # 避免框不可见：给极小宽高加一个最小值
        if width == 0:
            width = 0.5
            xmin -= 0.25
        if height == 0:
            height = 0.5
            ymin -= 0.25

        rect = Rectangle(
            (xmin, ymin), width, height,
            edgecolor='blue', facecolor='none',
            linestyle='--', linewidth=1
        )
    result = PathInfo(path_list,rect,None,None,None)
    return result

def is_overlap(rect1, rect2):
    x1_min,y1_min = rect1.get_xy()
    w1 = rect1.get_width()
    h1 = rect1.get_height()
    x1_max,y1_max = x1_min+w1,y1_min+h1
    x2_min,y2_min = rect2.get_xy()
    w2 = rect2.get_width()
    h2 = rect2.get_height()
    x2_max, y2_max = x2_min + w2, y2_min + h2

    if x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min:
        return False
    return True

def plot_box_pathlist(pathlist):
    all_x_min, all_y_min = [], []
    all_x_max, all_y_max = [], []

    # 遍历 pathlist 中所有 path
    for path, attr in pathlist:
        all_x_temp, all_y_temp = [], []
        # 画出每条 path
        for segment in path:
            try:
                start = segment.start
                end = segment.end
            except AttributeError:
                start = segment['start']
                end = segment['end']
            xs = [start.real, end.real]
            ys = [start.imag, end.imag]
            # ax.plot(xs, ys, color='black', linewidth=0.5)  # 画路径线

            all_x_temp.extend(xs)
            all_y_temp.extend(ys)

        if all_x_temp and all_y_temp:
            xmin, xmax = min(all_x_temp), max(all_x_temp)
            ymin, ymax = min(all_y_temp), max(all_y_temp)
            all_x_min.append(xmin)
            all_x_max.append(xmax)
            all_y_min.append(ymin)
            all_y_max.append(ymax)

    # 计算整体外接矩形
    x_min = min(all_x_min)
    x_max = max(all_x_max)
    y_min = min(all_y_min)
    y_max = max(all_y_max)
    width = x_max - x_min
    height = y_max - y_min

    # 添加矩形框
    rect = Rectangle(
        (x_min, y_min), width, height,
        edgecolor='blue', facecolor='none',
        linestyle='--', linewidth=1
    )

    result = PathInfo(pathlist,rect,((x_min + x_max) / 2, (y_max + y_min) / 2),width,height)
    return result

# 第二次处理的第二步筛选函数
def second_filter(stringlists,max_num_string):
    result_group = []
    result0 = False
    for stringlist in stringlists:
        if len(stringlist.paths) < max_num_string +3:
            if len(stringlist.centerpoints) >= 2:
                cx0,cy0 = stringlist.centerpoints[0]
                cx1,cy1 = stringlist.centerpoints[1]
                angle = math.atan2((cy1-cy0),(cx1-cx0))/3.1415926*180
            else:
                angle = 0.0
            result_group.append(StrAndAngle(stringlist.paths, angle))

    return result_group

# 判断一个 path 对象是否全由 Line 段构成（即是多段线）
def is_polyline_path(path_tuple):
    path_data = path_tuple[1].get('d')  # 从属性中提取 d 值
    if not path_data:
        return False
    path = parse_path(path_data)
    for segment in path:
        if not isinstance(segment, Line):
            return False
    return True

# 最小面积的外接矩形框：
def min_area_bounding_rect(path_list):
    all_points = []
    for patht in path_list:
        # 收集所有路径中的坐标点
        path, attr = patht
        for segment in path:
            all_points.append([segment.start.real, segment.start.imag])
            all_points.append([segment.end.real, segment.end.imag])

        if not all_points:
            # print("没有找到任何点。")
            return

    points_np = np.array(all_points, dtype=np.float32)

    # 计算最小面积外接矩形
    rect = cv2.minAreaRect(points_np)  # ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # 解包矩形参数
    (cx, cy), (w, h), angle = rect

    return (w, h, (cx, cy), angle)

def trans_coord(x,y,attr):
    transform_str = attr.get('transform', '')

    # matrix 变换
    matrix_match = re.search(
        r'matrix\(\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+),\s*([\d\.\-eE]+)\s*\)',
        transform_str)
    if matrix_match:
        a, b, c, d, e, f = map(float, matrix_match.groups())
        x1 = a * x + c * y + e
        y1 = b * x + d * y + f
        return (x1,y1)

    # scale 变换
    scale_match = re.search(r'scale\(\s*([\d\.\-eE]+)(?:\s*,\s*([\d\.\-eE]+))?\s*\)', transform_str)
    if scale_match:
        sx = float(scale_match.group(1))
        sy = float(scale_match.group(2)) if scale_match.group(2) else sx
        return (x * sx, y * sy)

    # 无变换
    return (x, y)

def cal_time(window,sw, group):
    path_group1 = first_pretreatment(group, window.max_length_val)
    A, B, C = second_pretreatment(path_group1,
                                  window.str_length_val)

    # 合并 A, B, C
    all_paths = A + B + C

    window._total_task_count += len(all_paths)
    return all_paths


import multiprocessing
from functools import partial

def process_stroke_width_group(window,sw, group,progress_callback=None,stop_flag=None):
    path_group1 = first_pretreatment(group, window.max_length_val)
    A, B, C = second_pretreatment(path_group1,window.str_length_val)

    # 合并 A, B, C
    all_paths = A + B + C

    # 平均分成3份
    def chunk_list(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

    path_chunks = chunk_list(all_paths, 20)

    engine = RapidOCR(
        params={
            "Global.with_torch": True,
            "EngineConfig.torch.use_cuda": True,  # 使用torch GPU版推理
            "EngineConfig.torch.gpu_id": 0,  # 指定GPU id
        }
    )

    ocr_list = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for chunk in path_chunks:
            if stop_flag is not None and not stop_flag():
                break
            futures.append(executor.submit(create_images_ocr, chunk, engine = engine, stop_flag=stop_flag, progress_callback=progress_callback))

        for future in futures:
            if stop_flag is not None and not stop_flag():
                break
            ocr_list.extend(future.result())

    return ocr_list



from concurrent.futures import ThreadPoolExecutor

def parallel_ocr_by_stroke_width(window,groups,progress_callback=None,stop_flag=None):
    results = []

    with (ThreadPoolExecutor(max_workers=4) as executor):  # 线程数根据任务量和显存定
        futures = []
        for sw, group in groups.items():
            cal_time(window,sw,group)
        for sw, group in groups.items():
            futures.append(executor.submit(process_stroke_width_group,window, sw, group,progress_callback=progress_callback,stop_flag=stop_flag))

        for future in futures:
            results.extend(future.result())

    return results



import os
import sys
def resource_path(relative_path):
    """PyInstaller 资源路径适配"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def resource_path1(filename):
    """优先读取exe同目录下的文件，否则读取临时目录或当前目录"""
    if hasattr(sys, '_MEIPASS'):
        base_path = os.path.dirname(sys.executable)  # exe 所在目录
    else:
        base_path = os.path.abspath(".")  # 开发时目录

    return os.path.join(base_path, filename)


class ObbInfo:
    def __init__(self,polygon_patch = None, rect = None,box=None):
        self.polygon_patch = polygon_patch #
        self.rect = rect # center或x,y = rect[0], width, height = rect[1], theta = rect[2]
        self.box = box

def plot_obb(pathlist):
    all_points = []

    # 遍历 pathlist 中所有 path
    for path, attr in pathlist:
        for segment in path:
            try:
                start = segment.start
                end = segment.end
            except AttributeError:
                start = segment['start']
                end = segment['end']
            xs = [start.real, end.real]
            ys = [start.imag, end.imag]

            all_points.extend(list(zip(xs, ys)))

    if not all_points:
        return None  # 无有效点则返回空

    # 将所有点转为 numpy 格式，注意 OpenCV 需要 float32
    points_np = np.array(all_points, dtype=np.float32)

    # 调用 OpenCV 的最小外接矩形（OBB）
    rect = cv2.minAreaRect(points_np)  # 返回中心坐标、(宽, 高)、角度
    box = cv2.boxPoints(rect)          # 得到四个角点
    box = np.array(box, dtype=np.float32)


    # 用四边形来表示 OBB
    polygon_patch = Polygon(box, closed=True, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1)

    # 计算中心点、宽高（方便构建 PathInfo）
    center = rect[0]
    width, height = rect[1]

    if width < height:
        temp = width
        width = height
        height = temp

    rect1 = [center,(width,height)]


    result = ObbInfo(polygon_patch,rect1,box)
    return result

def get_rect_data(e):
    angle = e.angle
    data = plot_obb(e.strPath)
    centerpoint = data.rect[0]
    width, height = data.rect[1]
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = data.box
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    if 0 < angle <= 90:
        min_y = min(p[1] for p in points)
        candidates = [p for p in points if p[1] == min_y]
        startpoint = max(candidates, key=lambda p: p[0]) # y->min  x->max
        idx = points.index(startpoint)
        if angle != 90:
            if idx == 3:
                point = points[0]
            else:
                point = points[idx+1]
            angle = math.atan2((point[1]-startpoint[1]),(point[0]-startpoint[0]))/3.1415926*180
    elif 90 < angle <= 180:
        startpoint = max(points, key=lambda p: (p[0], p[1]))  # x->max  y->max
        idx = points.index(startpoint)
        if angle != 180:
            if idx == 3:
                point = points[0]
            else:
                point = points[idx+1]
            angle = math.atan2((point[1]-startpoint[1]),(point[0]-startpoint[0]))/3.1415926*180
    elif -180 < angle <= -90:
        max_y = max(p[1] for p in points)
        candidates = [p for p in points if p[1] == max_y]
        startpoint = min(candidates, key=lambda p: p[0])  # y->max  x->min
        idx = points.index(startpoint)
        if angle != -90:
            if idx == 3:
                point = points[0]
            else:
                point = points[idx + 1]
            angle = math.atan2((point[1] - startpoint[1]), (point[0] - startpoint[0])) / 3.1415926 * 180
    else:
        startpoint = min(points, key=lambda p: (p[0], p[1]))  # x->min  y->min
        idx = points.index(startpoint)
        if angle != 0:
            if idx == 3:
                point = points[0]
            else:
                point = points[idx + 1]
            angle = math.atan2((point[1] - startpoint[1]), (point[0] - startpoint[0])) / 3.1415926 * 180

    return startpoint,centerpoint,width,height,angle

# def create_inter_image(groups,resize = (5,5),dpi = 300,padding=300):
#     pil_images = []  # ❷ 如需直接存 PIL 图像
#     INFO = []
#     for idx, e in enumerate(groups):
#         # --- 计算边界框 -------------------------------------------------
#         all_x, all_y = [], []
#         for path, attr in e.strPath:
#             for seg in path:
#                 all_x.extend([seg.start.real, seg.end.real])
#                 all_y.extend([seg.start.imag, seg.end.imag])
#
#         if not all_x:  # 跳过空组
#             continue
#
#         min_x, max_x = min(all_x), max(all_x)
#         min_y, max_y = min(all_y), max(all_y)
#
#         # --- 作图 -------------------------------------------------------
#         fig, ax = plt.subplots(figsize=(4,4), dpi=300)  # 每次新建
#         for path, attr in e.strPath:
#             for seg in path:
#                 xs = [seg.start.real, seg.end.real]
#                 ys = [seg.start.imag, seg.end.imag]
#                 ax.plot(xs, ys, color="black", linewidth=2)
#
#         ax.set_aspect("equal")
#         ax.invert_yaxis()
#         ax.axis("off")
#         ax.set_xlim(min_x - padding, max_x + padding)
#         ax.set_ylim(min_y - padding, max_y + padding)
#
#         # --- 保存到 BytesIO ---------------------------------------------
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", bbox_inches="tight",
#                     pad_inches=0, dpi=300)
#         buf.seek(0)  # 重新指向流头
#
#         # 或者转成 PIL.Image 放进另一列表（❷）
#         pil_img = Image.open(buf)
#         pil_images.append(pil_img.copy())  # copy，后面可安全关闭 buf
#
#         # --- 清理 -------------------------------------------------------
#         buf.close()  # 及时关闭 BytesIO
#         plt.close(fig)
#
#         startPoint, centerPoint, width, height, Angle = get_rect_data(e)
#         info = StringInfo(text=None, path=e.strPath, score=None, num=None, startPoint=startPoint,
#                               width=width, height=height, angle=Angle, centerpoint=centerPoint)
#         INFO.append(info)
#         print(idx)
#
#     return pil_images,INFO
#
# def ocr_inter_image(pil_images,info):
#     engine = RapidOCR(params={
#         "Global.with_torch": True,
#         "EngineConfig.torch.use_cuda": True,
#         "EngineConfig.torch.gpu_id": 0,
#     })
#     def ocr_one(img, idx=None):
#         """
#         img : PIL.Image 或 bytes
#         idx : 可选，用来打印进度
#         """
#         if idx is not None:
#             print(f"Processing #{idx}")
#         # engine 支持直接接收 PIL.Image
#         return engine(img)
#
#     max_workers = 4  # 4–6 条线程就够了
#     with ThreadPoolExecutor(max_workers=max_workers) as pool:
#         # enumerate 只是为了打印序号，可省略
#         futures = [pool.submit(ocr_one, img, i)
#                    for i, img in enumerate(pil_images)]
#         results = [f.result() for f in futures]
#     string_info = []
#     for idx,result in enumerate(results):
#         a = float(result.scores[0]) if result.scores and result.scores[0] is not None else 0.0
#         if a > 0.8 and result.txts:
#             text = [re.sub(r'[^A-Z0-9]', '', line) for line in result.txts]
#
#             temp = StringInfo(text=text, path=info[idx].path, score=result.scores, num=idx + 1, startPoint=info[idx].startPoint,
#                               width=info[idx].width, height=info[idx].height, angle=info[idx].angle, centerpoint=info[idx].centerpoint)
#             string_info.append(temp)
#
#     return string_info



if __name__ == '__main__':
    # paths,attributes = svg2paths('NX33.svg')
    #
    # # xmin, xmax = 176, 1472
    # # ymin, ymax = 296, 560 # NX19
    # # xmin, xmax = 135, 690
    # # ymin, ymax = 433, 613 # NX00
    # xmin,xmax = 160,1485
    # ymin,ymax = 346,574
    #
    # # 提取符合条件的路径
    # filtered_paths = []
    # for path, attr in zip(paths, attributes):
    #     transform_str = attr.get('transform', '')
    #     if is_path_in_region(path,attr, xmin, xmax, ymin, ymax):
    #         filtered_paths.append((path, attr))
    # filtered_paths1 = filter_remain_continuous_paths(filtered_paths)
    # # debugTool.show_path(filtered_paths)
    #
    # groups = group_paths_by_stroke_width(filtered_paths1)
    #
    # result = []
    # result = parallel_ocr_by_stroke_width(groups)


    paths,attributes = svg2paths('NX19.svg')

    xmin, xmax = 176, 1472
    ymin, ymax = 296, 560 # NX19
    # xmin, xmax = 135, 690
    # ymin, ymax = 433, 613 # NX00
    # xmin,xmax = 160,1485
    # ymin,ymax = 346,574 # NX33
    # xmin, xmax = 441,1237
    # ymin, ymax = 204,645 # NX32

    # 提取符合条件的路径
    filtered_paths = []
    for path, attr in zip(paths, attributes):
        transform_str = attr.get('transform', '')
        if is_path_in_region(path,attr, xmin, xmax, ymin, ymax):
            filtered_paths.append((path, attr))
    all_x = []
    all_y = []
    for path, attr in filtered_paths:
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
        # print(max_x)
        # print(min_y)
        # print(max_y)
        # print(min_y)

    min_x,max_x,min_y,max_y = get_transform_coordinate((min_x,max_x,min_y,max_y),attr)
    # print('_'*70)
    # print(max_x)
    # print(min_y)
    # print(max_y)
    # print(min_y)

    # filtered_paths1 = filter_remain_continuous_paths(filtered_paths)
    # # debugTool.show_path(filtered_paths)
    #
    # groups = group_paths_by_stroke_width(filtered_paths1)
    #
    # engine = RapidOCR(
    #     params={
    #         "Global.with_torch": True,
    #         "EngineConfig.torch.use_cuda": True,  # 使用torch GPU版推理
    #         "EngineConfig.torch.gpu_id": 0,  # 指定GPU id
    #     }
    # )
    #
    # result = []
    # for sw, group in groups.items():
    #     # debugTool.show_path(group)
    #     print(f'线宽为：{sw}')
    #     path_group1 = first_pretreatment(group,5000)
    #     # for idx,e in enumerate(path_group1):
    #     #     debugTool.show_path_str(e)
    #     A,B,C = second_pretreatment(path_group1,10)
    #     # debugTool.show_path_str(A)
    #     # debugTool.show_path_str(B)
    #     # debugTool.show_path_str(C)
    #     # print(len(A))
    #     # print(len(B))
    #     # print(len(C))
    #     ocr_list = A + B +C
    #     print(len(ocr_list))
    #     # for idx,e in enumerate(ocr_list):
    #     #     debugTool.show_path(e,save_path=f'output/{idx}.png',show=False)
    #     create_images_ocr(ocr_list,engine=engine)


