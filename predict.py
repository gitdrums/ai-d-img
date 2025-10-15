import os
from typing import Any
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import AutoImageProcessor, AutoModelForImageClassification

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the base CvT-13 architecture
        self.processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/cvt-13",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        # Load your trained weights from the baked-in path
        weights_path = "models/model_epoch_24.pth"
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define class labels
        self.labels = ["Real", "AI-Generated"]

    def predict(
        self,
        image: Path = Input(description="Input image to classify"),
    ) -> dict[str, Any]:
        """Run a single prediction on the model"""
        
        # Load and preprocess image
        img = Image.open(image).convert('RGB')
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        # Format results
        result = {
            "prediction": self.labels[predicted_class],
            "confidence": float(probabilities[0][predicted_class]),
            "probabilities": {
                "real": float(probabilities[0][0]),
                "ai_generated": float(probabilities[0][1])
            }
        }
        
        return result