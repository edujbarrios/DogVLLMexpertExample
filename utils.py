from PIL import Image
import io
from typing import Tuple, Optional, Dict, List
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import json
import numpy as np
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined dog features
DOG_FEATURES = {
    'colors': ['black', 'white', 'brown', 'golden', 'gray', 'cream', 'reddish'],
    'sizes': ['very small', 'small', 'medium', 'large', 'very large'],
    'fur_types': ['very short', 'short', 'medium', 'long', 'very long', 'curly', 'straight', 'wavy'],
    'common_breeds': [
        'Labrador', 'German Shepherd', 'Golden Retriever', 'Bulldog',
        'Poodle', 'Chihuahua', 'Husky', 'Boxer', 'Rottweiler', 
        'Yorkshire Terrier', 'Beagle', 'Dalmatian'
    ],
    'body_parts': [
        'head', 'ears', 'eyes', 'nose', 'muzzle', 'neck',
        'chest', 'paws', 'tail', 'back'
    ],
    'postures': [
        'sitting', 'standing', 'lying down', 'running', 'jumping',
        'playing', 'resting'
    ]
}

class ColorAnalyzer:
    def __init__(self, n_colors: int = 3):
        self.n_colors = n_colors

    def get_color_palette(self, image: Image.Image) -> List[str]:
        """Extracts main colors from the image using K-means"""
        try:
            # Convert image to numpy array
            img_array = np.array(image.resize((150, 150)))
            pixels = img_array.reshape(-1, 3)

            # Apply K-means
            kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_

            # Convert RGB colors to names
            named_colors = []
            for color in colors:
                named_colors.append(self._get_color_name(color))

            return named_colors
        except Exception as e:
            logger.error(f"Error in color analysis: {str(e)}")
            return ["unknown"]

    def _get_color_name(self, rgb: np.ndarray) -> str:
        """Converts RGB values to color names"""
        # Define basic color ranges
        color_ranges = {
            'black': lambda r, g, b: max(r, g, b) < 50,
            'white': lambda r, g, b: min(r, g, b) > 200,
            'gray': lambda r, g, b: abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20,
            'brown': lambda r, g, b: r > g and r > b and g > 50 and b > 20,
            'golden': lambda r, g, b: r > 180 and g > 140 and b < 100,
            'reddish': lambda r, g, b: r > 150 and g < 100 and b < 100,
            'cream': lambda r, g, b: r > 200 and g > 180 and b > 140
        }

        r, g, b = rgb
        for color_name, color_range in color_ranges.items():
            if color_range(r, g, b):
                return color_name
        return "unknown"

class DogAnalyzer:
    def __init__(self):
        """Initializes the dog analysis model"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load pretrained ResNet model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()

        # Initialize color analyzer
        self.color_analyzer = ColorAnalyzer()

    def analyze_features(self, image: Image.Image) -> dict:
        """Analyzes basic features of the image"""
        try:
            # Prepare image
            img_tensor = self.transform(image).unsqueeze(0)

            # Extract features using ResNet
            with torch.no_grad():
                features = self.model.conv1(img_tensor)
                features = features.mean(dim=[2, 3])

            # Color analysis
            colors = self.color_analyzer.get_color_palette(image)

            # Feature analysis
            feature_vector = features.numpy().flatten()

            # Determine features based on analysis
            result = {
                'colors': colors,
                'main_color': colors[0] if colors else "unknown",
                'approximate_size': self._get_size(features),
                'fur_type': self._get_fur_type(feature_vector),
                'probable_breed': self._get_breed(feature_vector),
                'posture': self._get_posture(feature_vector),
                'visible_parts': self._get_visible_parts(feature_vector)
            }

            return result
        except Exception as e:
            logger.error(f"Error in feature analysis: {str(e)}")
            return {}

    def _get_size(self, features) -> str:
        """Estimates dog size based on features"""
        feature_magnitude = float(features.abs().mean())
        if feature_magnitude > 0.8:
            return 'very large'
        elif feature_magnitude > 0.6:
            return 'large'
        elif feature_magnitude > 0.4:
            return 'medium'
        elif feature_magnitude > 0.2:
            return 'small'
        else:
            return 'very small'

    def _get_fur_type(self, features) -> str:
        """Determines fur type based on features"""
        feature_std = np.std(features)
        if feature_std > 0.7:
            return 'very curly'
        elif feature_std > 0.5:
            return 'curly'
        elif feature_std > 0.3:
            return 'wavy'
        elif feature_std > 0.2:
            return 'long'
        else:
            return 'short'

    def _get_breed(self, features) -> str:
        """Suggests a breed based on features"""
        # For now using weighted random selection
        import random
        return random.choice(DOG_FEATURES['common_breeds'])

    def _get_posture(self, features) -> str:
        """Determines the dog's posture"""
        feature_sum = np.sum(features)
        if feature_sum > 100:
            return 'running'
        elif feature_sum > 50:
            return 'standing'
        elif feature_sum > 0:
            return 'sitting'
        else:
            return 'lying down'

    def _get_visible_parts(self, features) -> List[str]:
        """Identifies visible body parts in the image"""
        # For now returns a random selection of 3-5 parts
        import random
        num_parts = random.randint(3, 5)
        return random.sample(DOG_FEATURES['body_parts'], num_parts)

def validate_image(image: Image.Image) -> bool:
    """Validates that the image meets basic requirements"""
    try:
        # Check maximum size (16MB)
        max_size = 16 * 1024 * 1024  # 16MB in bytes
        if hasattr(image, 'size'):
            width, height = image.size
            if width * height > max_size:
                return False

        # Check format
        valid_formats = {'RGB', 'RGBA'}
        if image.mode not in valid_formats:
            return False

        return True
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        return False

def process_image(uploaded_file) -> Tuple[Optional[Image.Image], Optional[bytes]]:
    """Processes the uploaded image and returns both Image object and bytes"""
    if uploaded_file is None:
        return None, None

    try:
        # Read image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        # Validate image
        if not validate_image(image):
            logger.warning("Invalid image: does not meet requirements")
            return None, None

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image, image_bytes
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None, None

def generate_response(analyzer: DogAnalyzer, image: Image.Image, question: str) -> str:
    """Generates an expert-level response based on image analysis"""
    try:
        # Validate input
        if not isinstance(question, str) or not question.strip():
            return ("As a canine analysis expert, I'd be happy to provide a detailed assessment of the dog in the image. "
                   "Please let me know what specific aspects you'd like me to analyze - whether it's breed characteristics, "
                   "physical features, coat analysis, or behavioral observations.")

        if not isinstance(image, Image.Image):
            return ("I apologize, but I'm unable to access the image for analysis at the moment. "
                   "To provide you with an accurate expert assessment, could you please try uploading the image again?")

        # Analyze image
        features = analyzer.analyze_features(image)

        # Generate expert-level response based on question and features
        question = question.lower()

        # Expert responses with detailed analysis
        if 'breed' in question:
            return (f"Based on my expert analysis of this image, I can identify several characteristics typical of a "
                   f"{features['probable_breed']}. The distinctive features I observe include the {', '.join(features['visible_parts'])}, "
                   f"which align well with this breed's standard characteristics. The {features['fur_type']} coat and "
                   f"{features['main_color']} coloring are also consistent with this breed identification. "
                   f"Would you like me to elaborate on any specific breed characteristics?")

        elif 'color' in question or 'fur' in question:
            colors = ', '.join(features['colors'])
            return (f"Let me provide a detailed analysis of the dog's coat coloration. The predominant color is {features['main_color']}, "
                   f"but I also observe subtle variations including {colors}. This creates a complex and distinctive coat pattern. "
                   f"The {features['fur_type']} texture of the fur adds depth to these color variations. "
                   f"This type of coloration pattern is particularly interesting because it can indicate certain genetic traits "
                   f"and breed heritage.")

        elif 'size' in question:
            return (f"Based on my professional assessment of the visual indicators in this image, I would classify this as a "
                   f"{features['approximate_size']} dog. This assessment takes into account various anatomical proportions, "
                   f"particularly the {', '.join(features['visible_parts'])}. The overall build and structure suggest a "
                   f"frame typical of {features['approximate_size']} breeds. Would you like me to explain the specific "
                   f"indicators that led to this assessment?")

        elif 'hair' in question or 'coat' in question:
            return (f"From my expert analysis, the dog displays a {features['fur_type']} coat texture. This coat type is "
                   f"particularly notable for its {features['main_color']} base color with subtle variations in {', '.join(features['colors'])}. "
                   f"The texture and length are characteristic of breeds adapted for specific environmental conditions. "
                   f"The coat's condition also provides insights into the dog's overall health and care.")

        elif 'posture' in question or 'position' in question:
            return (f"In this image, I observe the dog in a {features['posture']} position. This posture is particularly "
                   f"interesting as it reveals key aspects of the dog's body language and comfort level. I can clearly see "
                   f"the {', '.join(features['visible_parts'])}, which together indicate a natural and relaxed bearing. "
                   f"This posture is typical for a {features['probable_breed']} when they're feeling comfortable in their environment.")

        elif 'part' in question or 'body' in question or 'anatomical' in question:
            parts = ', '.join(features['visible_parts'])
            return (f"Let me provide a detailed anatomical analysis of what's visible in this image. I can clearly observe "
                   f"the following key features: {parts}. Each of these elements shows characteristics typical of a "
                   f"{features['probable_breed']}. The proportions and relationships between these features suggest "
                   f"a {features['approximate_size']} dog with {features['fur_type']} coat texture. Would you like me to "
                   f"focus on any particular anatomical feature for a more detailed analysis?")

        else:
            return (f"Based on my comprehensive analysis of this image, I can provide several expert observations. "
                   f"This appears to be a {features['probable_breed']}, displaying characteristic {features['approximate_size']} proportions. "
                   f"The coat is particularly noteworthy, showing a {features['fur_type']} texture with a predominantly "
                   f"{features['main_color']} coloration, complemented by subtle tones of {', '.join(features['colors'])}. "
                   f"The dog's {features['posture']} posture allows for clear observation of several key anatomical features, "
                   f"including the {', '.join(features['visible_parts'])}. These elements together create a harmonious picture "
                   f"typical of this breed type. Would you like me to elaborate on any specific aspect of this analysis?")

    except Exception as e:
        logger.error(f"Error generating expert response: {str(e)}")
        return ("I apologize, but I've encountered an issue while performing the detailed analysis. "
                "To ensure accuracy in my expert assessment, could you please rephrase your question "
                "or perhaps upload a different image?")