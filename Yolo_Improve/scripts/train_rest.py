import sys
sys.path.append('..')  # 添加上级父目录到搜索路径

from ultralytics import YOLO

# pip list --format=freeze > requirements.txt

if __name__ == "__main__":
    model = YOLO("yolov8m-pose-bifpn.yaml")
    model.load("best.pt")  # 加载初始化权重
    model.train(
        data="self_datas_pose.yaml",
        imgsz=640,
        epochs=8,
        batch=16,
        optimizer="SGD",
        patience=0, # 关闭近x个训练没原始高后停止训练
    )