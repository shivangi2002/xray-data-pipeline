import os
from PIL import Image
from torchvision import transforms
class XRayDatasetDefinition:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
       
       # Create a mapping from class names to integer labels
        self.class_to_label = {class_name: idx 
            for idx, class_name in enumerate(sorted(os.listdir(dataset_path)))
            }
        
        self.transform = transforms.Resize((224, 224))
        # Create a list of (image_path, label) tuples
        self.samples = []
        for class_name in self.class_to_label:
            label = self.class_to_label[class_name]
            class_path = os.path.join(self.dataset_path,class_name)
            
            for img_name in sorted(os.listdir(class_path)):
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, label))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        img  = self.transform(img)
        return img, label