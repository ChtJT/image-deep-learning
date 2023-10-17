import os
import shutil
import random

def split_data_for_pretraining(base_path="data_split"):
    # Directories for train, pretrain and classes
    train_dir = os.path.join(base_path, "train")
    pretrain_dir = os.path.join(base_path, "pretrain")
    class_names = ["pos", "mid", "nag"]

    # Create pretrain directory if it doesn't exist
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    for class_name in class_names:
        # Create pretrain class directory
        pretrain_class_dir = os.path.join(pretrain_dir, class_name)
        if not os.path.exists(pretrain_class_dir):
            os.makedirs(pretrain_class_dir)

        # Get list of images for the class
        train_class_dir = os.path.join(train_dir, class_name)
        images = [img for img in os.listdir(train_class_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)  # Shuffle the list of images

        # Split 80% for pretraining
        split_point = int(0.8 * len(images))
        for img in images[:split_point]:
            # Move the image from train to pretrain
            shutil.move(os.path.join(train_class_dir, img), os.path.join(pretrain_class_dir, img))

    return "Data split for pretraining completed!"

# Execute the data split function
split_data_for_pretraining(base_path="model/data_split")
