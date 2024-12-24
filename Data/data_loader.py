import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from tensorflow.keras.utils import to_categorical # type: ignore

@dataclass
class ImageInfo:
    filename: str
    width: int
    height: int
    objects: List[Dict[str, Any]]

@dataclass
class FoodData:
    volume: float
    weight: float
    density: float
    energy: Optional[float]

class ECUSTFDDataLoader:
    def __init__(self, base_path: str, target_size: Tuple[int, int] = (224, 224)):
        self.base_path = Path(base_path)
        self.target_size = target_size
        self.density_map: Dict[str, FoodData] = {}
        self._load_density_data()

    def _load_density_data(self) -> None:
        """Load and preprocess density information from Excel file."""
        density_path = self.base_path / 'density.xls'
        if not density_path.exists():
            raise FileNotFoundError(f"Density file not found: {density_path}")

        density_df = pd.read_excel(density_path)
        density_df.columns = [col.lower().strip() for col in density_df.columns]
        density_df['type'] = density_df['type'].str.lower()

        for _, row in density_df.iterrows():
            food_type = row['type'].lower()
            volume = float(row['volume(mm^3)'])
            weight = float(row['weight(g)'])
            self.density_map[food_type] = FoodData(
                volume=volume,
                weight=weight,
                density=weight / volume,
                energy=float(row['energy(kcal/g)']) if 'energy(kcal/g)' in row else None
            )

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image to ensure consistent size and format."""
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        return np.array(image) / 255.0

    def get_paired_images(self, img_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get top and side view image pairs."""
        top_path = self.base_path / 'Images' / f'{img_id}T.jpg'
        side_path = self.base_path / 'Images' / f'{img_id}S.jpg'

        if not all(p.exists() for p in [top_path, side_path]):
            img_path = self.base_path / 'Images' / f'{img_id}.jpg'
            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                return self.preprocess_image(img), None
            return None, None

        top_img = Image.open(top_path).convert('RGB')
        side_img = Image.open(side_path).convert('RGB')

        return (
            self.preprocess_image(top_img),
            self.preprocess_image(side_img)
        )

    def load_split(self, split: str = 'train') -> Dict[str, Any]:
        """Load dataset split with available views and annotations."""
        split_file = self.base_path / 'ImageSets' / 'Main' / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            image_ids = [x.strip() for x in f.readlines() if x.strip()]

        dataset = {
            'images': [], 'labels': [], 'bboxes': [],
            'volumes': [], 'weights': [], 'densities': [],
            'energies': [], 'image_ids': []
        }

        skipped: List[str] = []
        
        for img_id in image_ids:
            try:
                top_img, side_img = self.get_paired_images(img_id)
                if top_img is None:
                    skipped.append(f"Missing image for {img_id}")
                    continue

                ann_path = self.base_path / 'Annotations' / f'{img_id}.xml'
                if not ann_path.exists():
                    ann_path = self.base_path / 'Annotations' / f'{img_id}T.xml'
                    if not ann_path.exists():
                        skipped.append(f"Missing annotation for {img_id}")
                        continue

                ann_info = self.parse_annotation(ann_path)
                food_objects = [obj for obj in ann_info.objects if obj['name'] != 'coin']

                if not food_objects:
                    skipped.append(f"No food objects in {img_id}")
                    continue

                food_type = food_objects[0]['name']
                if food_type not in self.density_map:
                    skipped.append(f"No density data for {food_type}")
                    continue

                food_data = self.density_map[food_type]
                dataset['images'].append(top_img)  # Now already preprocessed and normalized
                dataset['labels'].append(food_type)
                dataset['bboxes'].append(food_objects[0]['bbox'])
                dataset['volumes'].append(food_data.volume)
                dataset['weights'].append(food_data.weight)
                dataset['densities'].append(food_data.density)
                dataset['energies'].append(food_data.energy)
                dataset['image_ids'].append(img_id)

            except Exception as e:
                skipped.append(f"Error processing {img_id}: {str(e)}")
                continue

        # Convert lists to numpy arrays
        dataset['images'] = np.stack(dataset['images'])  # This should work now as all images are preprocessed
        for key in ['labels', 'volumes', 'weights', 'densities', 'energies']:
            dataset[key] = np.array(dataset[key])

        print(f"\nLoaded {len(dataset['labels'])} images for {split} split")
        if skipped:
            print(f"Skipped {len(skipped)} items. First few reasons:")
            for reason in skipped[:5]:
                print(f"- {reason}")

        return dataset

    def parse_annotation(self, annotation_path: Path) -> ImageInfo:
        """Parse VOC-style annotation XML file."""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        size = root.find('size')
        if size is None or root.find('filename') is None:
            raise ValueError(f"Invalid annotation file format: {annotation_path}")

        image_info = ImageInfo(
            filename=root.find('filename').text,
            width=int(size.find('width').text),
            height=int(size.find('height').text),
            objects=[]
        )

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            if bbox is None or obj.find('name') is None:
                continue

            object_info = {
                'name': obj.find('name').text.lower(),
                'bbox': [
                    int(bbox.find(coord).text)
                    for coord in ['xmin', 'ymin', 'xmax', 'ymax']
                ]
            }
            image_info.objects.append(object_info)
        
        return image_info

    def prepare_data(self) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Prepare training and validation data with one-hot encoded labels."""
        train_data = self.load_split('train')
        val_data = self.load_split('val')

        if len(train_data['labels']) == 0:
            raise ValueError("No training labels found. Please check the dataset.")
        if len(val_data['labels']) == 0:
            raise ValueError("No validation labels found. Please check the dataset.")

        all_labels = np.concatenate([train_data['labels'], val_data['labels']])
        unique_labels = np.unique(all_labels)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        for split_data in [train_data, val_data]:
            labels_encoded = np.array([label_to_index[label] for label in split_data['labels']])
            split_data['labels_onehot'] = to_categorical(labels_encoded, num_classes=len(unique_labels))

        return train_data, val_data, unique_labels

def main():
    base_path = r'C:\Users\salah\OneDrive\Bureau\Food Classification Project\Data\ECUSTFD-resized-'
    
    try:
        loader = ECUSTFDDataLoader(base_path)
        train_data, val_data, unique_labels = loader.prepare_data()

        print("\nDataset Statistics:")
        print(f"Training samples: {len(train_data['labels'])}")
        print(f"Validation samples: {len(val_data['labels'])}")
        print(f"Number of classes: {len(unique_labels)}")
        print(f"Classes: {', '.join(unique_labels)}")

        print("\nAverage measurements (training set):")
        print(f"Volume: {np.mean(train_data['volumes']):.2f} mm³")
        print(f"Weight: {np.mean(train_data['weights']):.2f} g")
        print(f"Density: {np.mean(train_data['densities']):.4f} g/mm³")
        if train_data['energies'][0] is not None:
            print(f"Energy: {np.mean(train_data['energies']):.2f} kcal/g")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()