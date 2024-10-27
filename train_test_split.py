import os
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

def load_json(file_path):
    with open(file_path, "r") as file_ptr:
        return json.load(file_ptr)

def save_json(data, file_path):
    with open(file_path, "w") as file_ptr:
        json.dump(data, file_ptr)

def copy_image(src, dst):
    shutil.copy(src, dst)

def main():
    image_dir_path = os.path.join("Dataset", "Pedestrian_dataset")
    ground_truth_file = os.path.join("Dataset", "ground_truth.json")

    coco_dict = load_json(ground_truth_file)
    pprint(coco_dict.keys())
    # Output: ['images', 'annotations', 'categories']

    images = coco_dict["images"]
    annotations = coco_dict["annotations"]
    categories = coco_dict["categories"]

    split = 0.8
    random.shuffle(images)
    split_index = int(len(images) * split)
    train_images = images[:split_index]
    validation_images = images[split_index:]

    train_image_ids = {image["id"] for image in train_images}
    validation_image_ids = {image["id"] for image in validation_images}

    train_annotations = [ann for ann in annotations if ann["image_id"] in train_image_ids]
    validation_annotations = [ann for ann in annotations if ann["image_id"] in validation_image_ids]

    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }

    validation_data = {
        "images": validation_images,
        "annotations": validation_annotations,
        "categories": categories
    }

    os.makedirs("processed_dataset", exist_ok=True)
    os.makedirs(os.path.join("processed_dataset", "annotations"), exist_ok=True)
    train_annotations_file = os.path.join("processed_dataset", "annotations", "instances_train2017.json")
    validation_annotations_file = os.path.join("processed_dataset", "annotations", "instances_val2017.json")

    save_json(train_data, train_annotations_file)
    save_json(validation_data, validation_annotations_file)

    train_image_dir = os.path.join("processed_dataset", "train2017")
    validation_image_dir = os.path.join("processed_dataset", "val2017")
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(validation_image_dir, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        for image in train_images:
            src = os.path.join(image_dir_path, image["file_name"])
            dst = os.path.join(train_image_dir, image["file_name"])
            executor.submit(copy_image, src, dst)

        for image in validation_images:
            src = os.path.join(image_dir_path, image["file_name"])
            dst = os.path.join(validation_image_dir, image["file_name"])
            executor.submit(copy_image, src, dst)

    print("Dataset has been created")

if __name__ == "__main__":
    main()