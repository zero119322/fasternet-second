from ultralytics import YOLO
if __name__ == '__main__':
    # 加载模型
    model = YOLO("ultralytics/cfg/models/v8/yolov8l.yaml")  # 从头开始构建新模型
    print(model.model)

    # Use the model
    results = model.train(data="VOC1.yaml", epochs=300, device='0', batch=8,seed=42,patience=10.pretrained=False)  # 训练模型
