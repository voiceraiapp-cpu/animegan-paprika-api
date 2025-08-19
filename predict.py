import tempfile
from typing import Optional
import torch
from PIL import Image
import cog

class Predictor(cog.Predictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("🔄 Loading AnimeGAN Paprika model...")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        
        try:
            # Load paprika model
            self.paprika_model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "generator",
                pretrained="paprika",
                force_reload=False
            ).to(self.device).eval()
            
            # Load face2paint utility
            self.face2paint = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "face2paint",
                size=512,
                force_reload=False
            )
            
            print("✅ Paprika model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def predict(
        self,
        image: cog.Path,
        style_strength: float = cog.Input(
            description="Style strength (higher = more anime style)",
            default=1.0,
            ge=0.1,
            le=2.0
        )
    ) -> cog.Path:
        """
        Convert image to anime/cartoon style using AnimeGAN Paprika model
        """
        
        try:
            # Load and process input image
            print("📸 Processing input image...")
            input_image = Image.open(str(image)).convert("RGB")
            
            # Apply paprika style transformation
            print("🎨 Applying Paprika anime style...")
            with torch.no_grad():
                # Use face2paint for full scene stylization
                styled_image = self.face2paint(
                    self.paprika_model, 
                    input_image
                )
            
            # Optional: Adjust style strength by blending
            if style_strength != 1.0:
                print(f"⚖️ Adjusting style strength: {style_strength}")
                styled_image = Image.blend(
                    input_image, 
                    styled_image, 
                    style_strength
                )
            
            # Save output to temporary file
            output_path = tempfile.mktemp(suffix=".png")
            styled_image.save(output_path, "PNG")
            
            print("✅ Anime stylization completed!")
            return cog.Path(output_path)
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            raise e
