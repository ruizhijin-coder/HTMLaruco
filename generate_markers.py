import cv2
import numpy as np

def save_marker_png_cv2(marker_id, filename):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # 获取 6x6 的原始位图 (0=黑, 255=白)
    img_6x6 = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)
    
    # 放大倍率
    scale = 100
    size = 6 * scale
    
    # 创建 BGRA 四通道图像（B, G, R, Alpha）
    # 初始化全透明 (Alpha = 0)
    img_bgra = np.zeros((size, size, 4), dtype=np.uint8)
    
    for r in range(6):
        for c in range(6):
            if img_6x6[r, c] == 0: # 黑色色块
                # 设置 BGR 为 0,0,0（黑色），Alpha 为 255（不透明）
                img_bgra[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = [0, 0, 0, 255]
            else:
                # 白色色块设置为透明 (Alpha = 0)
                # 其实默认就是 0，这里显式写出来方便理解
                img_bgra[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = [0, 0, 0, 0]
                
    # 使用 OpenCV 保存 PNG (Opencv 自动按通道识别 alpha)
    cv2.imwrite(filename, img_bgra)
    print(f"Saved Transparent PNG (via OpenCV): {filename}")

if __name__ == "__main__":
    ids = [1, 2, 3]
    save_marker_png_cv2(ids[0], 'marker_top_left.png')
    save_marker_png_cv2(ids[1], 'marker_top_right.png')
    save_marker_png_cv2(ids[2], 'marker_bottom_right.png')
