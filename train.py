from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Disable all data augmentation
results = model.train(
    data="data2/data.yaml",
    epochs=1000,
    imgsz=2000,
    batch=1,
    patience=50,
    max_det=3000,
    augment=False,   # Disable augmentations globally

    # Explicitly disable individual augmentations

    translate=0.0,   # Disable translation
    scale=0.0,       # Disable scaling
    shear=0.0,       # Disable shear
    perspective=0.0, # Disable perspective transform
    mosaic=0.0,      # Disable mosaic augmentation
    mixup=0.0,       # Disable mixup augmentation
    copy_paste=0.0   # Disable copy-paste augmentation
)

# Validate the model
metrics = model.val()
print("mAP50-95:", metrics.box.map)
print("mAP50:", metrics.box.map50)
print("mAP75:", metrics.box.map75)
print("Per-class mAPs:", metrics.box.maps)
