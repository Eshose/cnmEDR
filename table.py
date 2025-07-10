import pandas as pd
import pytesseract
import os
import re
import fitz
import cv2
import edr2
import numpy as np

# 设置Tesseract OCR引擎的路径（需要根据本地安装路径修改）
# pytesseract.pytesseract.tesseract_cmd = r'E:\Code_Tool\Tesseract-OCR\tesseract.exe'  # 修改为你本机路径
tesseract_path = edr2.resource_path("Tesseract-OCR/tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path # 修改为你本机路径

def ocr_key_regions(window,pdf_path: str, page_number: int = 0, zoom: float = 8.0, extra_txt_path: str = None):
    """
    识别PDF上方关键区域的信息

    参数:
        pdf_path: PDF文件路径
        page_number: 要处理的页码
        zoom: 放大倍数，提高识别精度
        extra_txt_path: 输出文本文件路径
    """
    # 定义要识别的关键区域及其坐标 (左, 上, 右, 下)
    (x0, y0, w0, h0) = QrectF_return(window.rectItem0)
    (x1, y1, w1, h1) = QrectF_return(window.rectItem1)
    (x2, y2, w2, h2) = QrectF_return(window.rectItem2)
    (x3, y3, w3, h3) = QrectF_return(window.rectItem3)


    regions = {
        "材料规格": (x0/window.tran_x, y0/window.tran_y, (x0+w0)/window.tran_x, (y0+h0)/window.tran_y),
        "材质": (x1/window.tran_x, y1/window.tran_y, (x1+w1)/window.tran_x, (y1+h1)/window.tran_y),
        "切割指令号": (x2/window.tran_x, y2/window.tran_y, (x2+w2)/window.tran_x, (y2+h2)/window.tran_y),
        "比例": (x3/window.tran_x, y3/window.tran_y, (x3+w3)/window.tran_x, (y3+h3)/window.tran_y)
    }


    # 获取文件名（无扩展名）
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # 若未指定 extra_txt_path，则自动命名
    if extra_txt_path is None:
        extra_txt_path = os.path.join(".", f"{pdf_name}_extra.txt")
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    matrix = fitz.Matrix(zoom, zoom)# 创建比例放大矩阵

    results = []

    # 遍历每个关键区域进行识别
    for label, coords in regions.items():
        rect = fitz.Rect(*coords)
        pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)# 获取区域图像
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 使用Tesseract进行OCR识别
        raw_text = pytesseract.image_to_string(img_rgb, config='--oem 3 --psm 7', lang='num').strip()
        # raw_text = pytesseract.image_to_data(img_rgb, config='--oem 3 --psm 7', lang='num')
        #去除空格和换行符
        def clean_text(text):
            return re.sub(r'\s+', '', text)
        cleaned = clean_text(raw_text)# 清理识别结果

        # print(f"\n===== {label} =====")
        # print(raw_text)

        if label == "比例":
            results.append(f'{label}: 1:{cleaned}')
        else:
            results.append(f'{label}: {cleaned}')

    return results



def extract_pdf_data(
        window,
    pdf_path: str,
    page_number: int = 0,
    clip_rects: list = None,
    zoom: float = 8.0,
    output_dir: str = "."
):
    """
    提取PDF下方表格区域的数据

    参数:
        pdf_path: PDF文件路径
        page_number: 要处理的页码
        clip_rects: 要提取的区域列表
        zoom: 放大倍数
        output_dir: 输出目录
    """
    (x4, y4, w4, h4) = QrectF_return(window.rectItem4)
    (x5, y5, w5, h5) = QrectF_return(window.rectItem5)
    (x6, y6, w6, h6) = QrectF_return(window.rectItem6)
    (x7, y7, w7, h7) = QrectF_return(window.rectItem7)
    if clip_rects is None:
        #下表格区域
        clip_rects = [
            (x4/window.tran_x, y4/window.tran_y, (x4+w4)/window.tran_x, (y4+h4)/window.tran_y),   # 第一列
            (x5/window.tran_x, y5/window.tran_y, (x5+w5)/window.tran_x, (y5+h5)/window.tran_y), # 第二列
            (x6/window.tran_x, y6/window.tran_y, (x6+w6)/window.tran_x, (y6+h6)/window.tran_y), # 第三列
            (x7/window.tran_x, y7/window.tran_y, (x7+w7)/window.tran_x, (y7+h7)/window.tran_y), # 第四列
        ]

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_csv = os.path.join(output_dir, f"{pdf_name}_output.csv")

    all_data = []

    #选用的图像增强方法
    def image_enhance(image):

        # 第一步：去噪声（双边滤波，更好保护文字边缘）
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        # 第二步：增强对比度（自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        # 第三步：二值化处理（更适合 OCR）
        _, binary = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def extract_and_clean_pdf_region(pdf_path, page_number, clip_rect, zoom):
        # 提取PDF区域转图像并进行去线处理
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        rect = fitz.Rect(*clip_rect)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)

        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 2)
        # 定义水平和垂直核，用于检测表格线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        # 检测水平线和垂直线
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 50:
                cv2.drawContours(vertical_lines, [contour], -1, 0, -1)
        # 合并水平线和垂直线
        lines_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        lines_mask = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=1)
        # 使用修复算法去除线条
        result = cv2.inpaint(img, lines_mask, 3, cv2.INPAINT_TELEA)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # 进行图像增强
        result = image_enhance(result)
        return result

    def parse_ocr_lines(text):
        lines = text.strip().split('\n')
        result = []
        for line in lines:
            match = re.match(
                r'(\d+)[\.:S]*\s*C{1,2}\s+([0-9A-Z\-/\$]+)\s+\d+\s+([0-9%+*/]+)',
                line.strip(), re.IGNORECASE
            )
            if match:
                index, code, size = match.groups()
                result.append({
                    "序号": int(index),
                    "零件编码": code,
                    "尺寸": size
                })
            else:
                result.append({
                    "原始行": line.strip(),
                    "解析状态": "❌未能解析"
                })
        return result

        # 提取下方表格区域
    for idx, rect in enumerate(clip_rects):
        img = extract_and_clean_pdf_region(pdf_path, page_number, rect, zoom)
        #按编号保存临时图像
        temp_image_path = os.path.join(output_dir, f"{pdf_name}_temp_{idx}.png")
        cv2.imwrite(temp_image_path, img)
        ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6', lang='eng7')
        # ocr_data = pytesseract.image_to_data(img, config='--oem 3 --psm 6', lang='eng7')
        #置信度显示
        # print(ocr_data)
        # print(ocr_text)

        data = parse_ocr_lines(ocr_text)
        # filtered_data = [
        #     row for row in data
        #     if isinstance(row, dict) and any(v not in [None, '', '❌未能解析'] for v in row.values())
        # ]
        # if filtered_data:
        all_data.extend(data)

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    return  all_data


from PyQt5.QtGui import QPolygonF

def QrectF_return(rectItem):
    rect = rectItem.rect()
    points = [
        rect.topLeft(),
        rect.topRight(),
        rect.bottomLeft(),
        rect.bottomRight()
    ]
    mapped_points = [rectItem.mapToScene(p) for p in points]
    polygon = QPolygonF(mapped_points)
    scene_rect = polygon.boundingRect()
    return (scene_rect.left(), scene_rect.top(), scene_rect.width(), scene_rect.height())




if __name__ == "__main__":

    pdf_path = '103HC101HC-0028-NX19.pdf'
    ocr_key_regions(pdf_path, page_number=0,)
    extract_pdf_data(pdf_path, page_number=0, zoom=10)



