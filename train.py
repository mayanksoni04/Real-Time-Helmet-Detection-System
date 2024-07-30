from ultralytics import YOLO
# model = YOLO('yolov8n.yaml') # pass any model type
# results = model.train(data="data.yaml", epochs=50)
model = YOLO('best.pt') 
results = model.predict(source="man with helmet.jpg", save=True, stream=False, show=True)
print("Number of layers:", len(model.layers))