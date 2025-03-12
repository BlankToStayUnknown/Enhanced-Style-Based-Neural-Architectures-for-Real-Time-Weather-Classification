from torch.utils.data import Dataset
import json
import os
from PIL import Image

# -----------------------------------------------------------------------
# DATASET MULTI-TACHES
# -----------------------------------------------------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, data_json, classes_json, transform=None, search_folder=None):
        with open(data_json, 'r') as f:
            self.data = json.load(f)
        with open(classes_json, 'r') as f:
            self.classes = json.load(f)
        self.transform = transform
        self.search_folder = search_folder  # Si précisé, ce dossier contient directement les images
        self.samples = []
        self.class_to_idx = {}
        self.task_classes = {}

        # Construire la correspondance des classes avec des labels en minuscules
        for task, class_list in self.classes.items():
            self.task_classes[task] = class_list
            self.class_to_idx[task] = {cls_name.lower(): idx for idx, cls_name in enumerate(class_list)}

        # Construire la liste des échantillons
        for folder, images in self.data.items():
            for img_name, img_info in images.items():
                # Si search_folder est précisé, on construit le chemin complet en concaténant ce dossier et le nom de l'image
                if self.search_folder:
                    image_identifier = os.path.join(self.search_folder, os.path.basename(img_info['image_path']))
                else:
                    image_identifier = img_info['image_path']
                labels = {}
                for task in self.classes.keys():
                    if task in img_info:
                        label = img_info[task].lower()
                        if label in self.class_to_idx[task]:
                            labels[task] = self.class_to_idx[task][label]
                        else:
                            print(f"Warning: label '{label}' for task '{task}' not found in class_to_idx")
                            labels[task] = None  # Gestion des labels inconnus
                    else:
                        labels[task] = None  # Gestion des labels manquants
                self.samples.append((image_identifier, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_identifier, labels = self.samples[idx]
        # Si search_folder est précisé, image_identifier est déjà le chemin complet
        img_path = image_identifier
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Chemin de l'image '{img_path}' introuvable.")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, labels

