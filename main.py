from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("ultralytics/cfg/models/v8/yolov8l.yaml")  # 从头开始构建新模型
    print(model.model)

    # Use the model
    results = model.train(data="VOC1.yaml", epochs=100, device='cpu', batch=16,workers=0)  # 训练模型