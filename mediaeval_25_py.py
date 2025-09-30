from ultralytics import YOLO
import os
import shutil
from sklearn.model_selection import train_test_split
import zipfile
import csv

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# 1. Define your paths (CHANGE THESE TO MATCH YOUR GOOGLE DRIVE PATHS)
source_real_dir = "D:/Research/progan_train/0_real"#D:\Research\progan_train\airplane
source_fake_dir = "D:/Research/progan_train/1_fake"
base_target_dir = "D:/Research/progan_train/dataset_final_v1"  # This is where the new YOLO dataset will be created
'''
# 2. Create the new folder structure
train_image_dir = os.path.join(base_target_dir, "train", "images")
train_label_dir = os.path.join(base_target_dir, "train", "labels")
val_image_dir = os.path.join(base_target_dir, "val", "images")
val_label_dir = os.path.join(base_target_dir, "val", "labels")

# Create the directories
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 3. Function to process a class of images
def process_class(source_dir, class_id, train_ratio=0.8):

    # Get list of all image files
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Split into train and validation sets
    train_files, val_files = train_test_split(image_files, train_size=train_ratio, random_state=42)

    # Function to process a list of files
    def process_file_list(file_list, image_dest_dir, label_dest_dir):
        for image_file in file_list:
            # Copy image
            src_image_path = os.path.join(source_dir, image_file)
            dest_image_path = os.path.join(image_dest_dir, image_file)
            shutil.copy2(src_image_path, dest_image_path)

            # Create corresponding label file
            label_filename = os.path.splitext(image_file)[0] + '.txt'
            dest_label_path = os.path.join(label_dest_dir, label_filename)

            # Create the content for the label file
            # Using a bounding box that covers 90% of the image center
            label_content = f"{class_id} 0.5 0.5 0.9 0.9"

            # Write the label file
            with open(dest_label_path, 'w') as f:
                f.write(label_content)

    # Process the training and validation files
    print(f"Processing {len(train_files)} training images for class {class_id}")
    process_file_list(train_files, train_image_dir, train_label_dir)

    print(f"Processing {len(val_files)} validation images for class {class_id}")
    process_file_list(val_files, val_image_dir, val_label_dir)

# 4. Run the function for both classes
print("Processing REAL images (class 0)...")
process_class(source_real_dir, class_id=0)

print("\nProcessing FAKE images (class 1)...")
process_class(source_fake_dir, class_id=1)

print("\nDone! Dataset is ready for YOLO training.")
print(f"Training images: {len(os.listdir(train_image_dir))}")
print(f"Training labels: {len(os.listdir(train_label_dir))}")
print(f"Validation images: {len(os.listdir(val_image_dir))}")
print(f"Validation labels: {len(os.listdir(val_label_dir))}")

'''

from ultralytics import YOLO

# Load the pre-trained model (recommended)
model = YOLO("yolo11n.pt")  # Use "yolo11s.pt" for better accuracy

# Train the model
results = model.train(
    data="D:/Research/progan_train/data_fin_1.yaml", # Simple path to the YAML file we created
    epochs=50,    # Good starting point
    imgsz=640,
    batch=8,     # Reduce to 8 if you get memory errors
    device=0,     # Use GPU
    name='real_vs_fake',
    workers=0
)

# After the last training epoch finishes...
print("\nTraining complete. Now running final validation on the BEST model...")

# Load the weights that got the highest mAP50 during training (not necessarily from the last epoch)
best_model = YOLO('D:/Research/runs/detect/real_vs_fake5/weights/best.pt')

# Run a final, thorough validation to get the definitive results
final_metrics = best_model.val() # <- This produces the final validatio

#zip it!!
# Set up test image path (make sure your test set is here)
test_images_dir = "D:/Research/taska_test_R03hsaaV7P/taska_test"  # Replace with your test images directory
#D:\Research\taska_test_R03hsaaV7P
# Define output paths for predictions
output_csv_filename = "ResearchRangers_constrained.csv"  # This is the CSV you will generate

# 2. Inference on Test Set and Generate CSV
def generate_predictions(model, test_images_dir, output_csv_filename, threshold=0.5):
    predictions = []

    # Get all image filenames in the test directory
    image_filenames = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Iterate through each image and get predictions
    for image_filename in image_filenames:
        image_path = os.path.join(test_images_dir, image_filename)

        # Run inference on the image
        results = model.predict(source=image_path, conf=threshold, imgsz=640)

        # Extract the confidence score for the "synthetic" class (class 1)
        prob = results.pandas().xywh['confidence'][0]  # Confidence score for the predicted class

        # Classify as synthetic (1) or real (0) based on the threshold
        label = 1 if prob >= threshold else 0

        # Append the prediction result in the required format (image_id, prob, label, threshold)
        predictions.append([image_filename, prob, label, threshold])

    # Write predictions to a CSV file
    with open(output_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'prob', 'label', 'threshold'])  # Write header

        # Write each prediction
        for prediction in predictions:
            writer.writerow(prediction)

    print(f"Predictions saved to {output_csv_filename}")


# 3. Compress the CSV into a ZIP file for submission
def zip_submission(csv_filename):
    zip_filename = csv_filename.replace('.csv', '.zip')
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_filename)
    
    print(f"ZIP file created: {zip_filename}")


# 4. Run the inference and CSV generation
generate_predictions(model, test_images_dir, output_csv_filename, threshold=0.5)

# 5. Create the ZIP file for submission
zip_submission(output_csv_filename)


