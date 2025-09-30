import os
import csv
from ultralytics import YOLO

# Parameters
model_path = "D:/Muthu_Books/SSN/MediaEval Task 25/runs/runs/detect/real_vs_fake7/weights/best.pt"  # your trained model
test_folder = "D:/Muthu_Books/SSN/MediaEval Task 25/taska_test"                 # folder with test images
output_csv = "submission.csv"
threshold = 0.5  # global threshold

# Load model
model = YOLO(model_path)

# Collect results
rows = [("image_id", "prob", "label", "threshold")]

for img_name in os.listdir(test_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_folder, img_name)
        
        results = model.predict(img_path, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()     # confidence scores
            classes = boxes.cls.cpu().numpy()    # class indices
            best_idx = confs.argmax()

            prob = float(confs[best_idx])
            label = int(classes[best_idx])       # 0=real, 1=synthetic
        else:
            prob = 0.0
            label = 0   # default to "real" if nothing detected

        rows.append((img_name, prob, label, threshold))

# Save CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Saved predictions to {output_csv}")
