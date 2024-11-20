import sys
sys.path.append('..')  # 添加上级父目录到搜索路径

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("best-self.pt")
    model.predict(
        source="../create_self_data/test_datas/",
        imgsz=640,
        save=True,
    )