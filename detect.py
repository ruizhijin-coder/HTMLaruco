import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    # 初始化 ArUco 字典
    # 注意：如果你使用的是较旧的 OpenCV，API 可能是 aruco.Dictionary_get(aruco.DICT_4X4_50)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 打开摄像头 (0 通常是默认摄像头)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查连接。")
        return

    print("正在运行 ArUco 检测... 按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图提高检测效率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测标记
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # 绘制所有检测到的标记
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 存储关键点位置（中心点）
            points = {}
            for i in range(len(ids)):
                marker_id = ids[i][0]
                if marker_id in [1, 2, 3]:
                    # 计算标记中心
                    c = corners[i][0]
                    center = np.mean(c, axis=0).astype(int)
                    points[marker_id] = center
                    
                    # 添加文字标签
                    label = ""
                    if marker_id == 1: label = "Top-Left (ID 1)"
                    elif marker_id == 2: label = "Top-Right (ID 2)"
                    elif marker_id == 3: label = "Bottom-Right (ID 3)"
                    
                    cv2.putText(frame, label, (center[0], center[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 如果左上、右上、右下三个码都全了，我们可以预测左下并画出扫描框
            if all(k in points for k in [1, 2, 3]):
                p1, p2, p3 = points[1], points[2], points[3]
                
                # 向量计算第四个点 (Bottom-Left)
                # p4 = p1 + (p3 - p2)
                p4 = p1 + (p3 - p2)
                
                # 绘制预测的扫描范围（红色矩形框）
                pts = np.array([p1, p2, p3, p4], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 0, 255), 3)
                cv2.putText(frame, "Polaroid Scan Area", (p4[0], p4[1] + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 显示结果
        cv2.imshow('ArUco Scan Jig Tester', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
