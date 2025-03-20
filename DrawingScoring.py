import json
import os

from enum import Enum
import requests
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Tuple

import base64

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv('OPENAI_API_KEY')

Client = OpenAI(api_key=api_key)


def add_category(category_name: str) -> tuple[str, int]:
    """
    Manages art categories using a JSON file.
    
    Args:
        category_name: The name of the category to check/add.
        
    Returns:
        tuple: (category_name.lower(), score) where score is 1 if the category exists,
               and 10 if it was added as a new category.
    """
    category_name_lower = category_name.lower()
    categories_file = "art_categories.json"
    
    # Load existing categories from JSON; if file doesn't exist or is invalid, start with an empty list.
    try:
        with open(categories_file, 'r') as f:
            data = json.load(f)
            categories = data.get("categories", [])
    except (FileNotFoundError, json.JSONDecodeError):
        categories = []
    
    # Check if the category already exists (ignoring case)
    if any(category_name_lower == cat.lower() for cat in categories):
        # Category exists: return with a score of 1.
        return category_name_lower, 1
    
    # Category does not exist: add it, update JSON file, and return score of 10.
    categories.append(category_name_lower)
    with open(categories_file, 'w') as f:
        json.dump({"categories": categories}, f, indent=2)
    return category_name_lower, 10

class DrawingScore(BaseModel):
    Player_name: str
    Player_age: int
    ArtCategory: str
    ArtDetailScore: int
    BaseImagePath: str
    AnalyzingImagePath: str
    ArtScore: float
    NameOfDrawing: str

system_prompt = """
You are an expert drawing scoring AI. Your task is to analyze a drawn image in comparison to a base image and score it based on the number of meaningful details added by the user. The base image contain a template, while the drawn image includes the user's additions or modifications.

Follow these steps:
1. **Compare Images**: Examine the base image and the drawn image to identify elements that have been added or modified in the drawn image. Only details not present in the base image should be counted(Note: Dont count repeated detail for example if have 3 balls with different color count all as 1).
2. **Identify the Main Subject**: Determine the primary subject of the drawing and assign it a descriptive name (e.g., "Paint Palette with Brush and Easel").
3. **Determine the Category**: Classify the drawing into a category.
4. **Detect Meaningful Details**: Identify all meaningful details added in the drawn image. A meaningful detail includes:
   - **Distinct Objects**: Extract individual items.
   - **Parts of Objects**: Subcomponents of the Each exctracted individual items, if any Subcomponents is repeated count once.
   - **Elements within Compositions**: Individual features within a larger element, such as the sun, sky, field, or cloud in a landscape on a canvas and etc.
   - **Attributes**: Descriptive characteristics like color or material, included in the detail's name (e.g., "brown handle," "blue sky").
   - **Grouped Similar Items**: Multiple similar items can be grouped as one detail when appropriate (e.g., "paint colors on palette" for multiple colors, "easel legs" for multiple legs), unless they are distinctly identifiable as separate entities.
5. **Describe Details**: For each detail, provide a descriptive name that includes key attributes, such as "brown brush handle," "black brush bristles," "gray ferrule," "palette thumb hole," "paint colors on palette," "sun," "blue sky," "green field," "easel legs," or "white cloud."
6. **Count Details**: Count the total number of meaningful details. Each detail is worth 1 point.
7. **Set Response Status**: If you are confident in your analysis and have accurately identified the details, set "Response" to "Undrestood." If unsure or unable to fully analyze the image, set it to "Not Undrestood."
8. **Return Analysis**: Provide your analysis in the following JSON format:

{
    "Response": "Undrestood" or "Not Undrestood",
    "Name": "name_of_the_drawn_item",
    "detected_category": "category_name",
    "details": ["detail1", "detail2", "detail3", ...],
    "detail_count": number_of_details
}

### Guidelines for Detail Detection
- **Break Down Objects**: Divide objects into their smallest meaningful parts. For example:
  - A brush should be split into "brown handle," "gray ferrule," and "black bristles."
  - A palette should include "thumb hole" and "paint colors on palette" as separate details.
  - A canvas with a landscape should list each element, such as "sun," "blue sky," "green field," and "white cloud," individually.
- **Include Attributes**: Incorporate attributes like color into the detail's name (e.g., "brown brush handle" instead of just "handle").
- **Handle Grouping**: Group similar items when they function as a collective unit (e.g., "paint colors on palette" for multiple colors, "easel legs" for multiple legs), unless the drawing emphasizes them as distinct (e.g., individually colored legs).
- **Exclude Base Image Details**: Ensure no elements from the base image are counted unless modified in the drawn image.
- **Assume Blank Base if Unclear**: If the base image is not provided or appears blank, treat all elements in the drawn image as added details.

### Objective
Your goal is to maximize the detail count by identifying all distinct, meaningful additions in the drawn image, ensuring an accurate and comprehensive score. For example, instead of detecting just "paint brush," recognize its "brown handle," "gray ferrule," and "black bristles" as three separate details, aligning with the user's expectation of detecting 10 specific details like "دسته قلم قهوه‌ای رنگ" (brown brush handle), "فرچه قلموی مشکی رنگ" (black brush bristles), etc.
"""

def process_ai_response(response_content: str) -> tuple[float, str, int, str]:
    try:
        # Try to find JSON content within the response
        # Sometimes GPT might wrap the JSON in markdown or add extra text
        response_content = response_content.strip()
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_content[start_idx:end_idx]
            analysis = json.loads(json_str)
        else:
            print("No valid JSON found in response")
            return 0, "unknown", 0, "unknown"
        
        # Get category score using add_category
        category_name = analysis.get('detected_category', 'unknown')
        category, category_score = add_category(category_name)
        name_of_drawing = str(analysis.get('Name', 'unknown'))
        
        # Calculate detail score
        detail_score = int(analysis.get('detail_count', 0))

        # Total score is category score + detail score
        total_score = category_score + detail_score
        print(f"Total score: {total_score}, Category: {category}, Detail score: {detail_score}, Name of drawing: {name_of_drawing}")
        
        return total_score, category, detail_score, name_of_drawing
        
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error processing AI response: {e}")
        print(f"Raw response: {response_content}")
        return 0, "unknown", 0, "unknown"

def analyze_drawing(base_image_path: str, drawing_path: str) -> tuple[float, str, int, str]:
    """Analyze the drawing using AI and return score and category"""
    try:
        with open(base_image_path, "rb") as image_file:
            base64_base_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Read and encode the drawn image
        with open(drawing_path, "rb") as image_file:
            base64_drawing_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the messages list with both images in a single user message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Base image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_base_image}"}},
                {"type": "text", "text": "Drawing to analyze:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_drawing_image}"}}
            ]}
        ]
        
        response = Client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name
            messages=messages,
            temperature=0.9
        )
        
        # Get the AI's analysis
        ai_response = response.choices[0].message.content
        if not ai_response or not ai_response.strip():
            print("Received an empty response from the API.")
            return 0, "unknown", 0, "unknown"
            
        print("Raw AI response:", ai_response)
        
        # Process the response and get score
        return process_ai_response(ai_response)
        
    except Exception as e:
        print(f"Error in analyze_drawing: {e}")
        return 0, "unknown", 0, "unknown"

def Player_playing():
    Player_name = input("Enter Player Name: ")
    Player_age = int(input("Enter Player Age: "))
    
    # Base Image validation
    while True:
        BaseImagePath = input("Enter Base Image Path(e.g. BasePic1.png): ")
        full_base_path = "./Images/Base/" + BaseImagePath
        if os.path.exists(full_base_path):
            break
        print(f"Error: File '{full_base_path}' does not exist. Please try again.")
    
    # Drawing Image validation
    while True:
        AnalyzingImagePath = input("Enter Drawing Image Path for example DrawingPic1.png: ")
        full_analyzing_path = "./Images/Drawing/" + AnalyzingImagePath
        if os.path.exists(full_analyzing_path):
            break
        print(f"Error: File '{full_analyzing_path}' does not exist. Please try again.")
    
    DrawingInfo = {
        "Player_name": Player_name,
        "Player_age": Player_age,
        "BaseImagePath": full_base_path,
        "Image_path": full_analyzing_path
    }
    
    score, category,detail_score,name_of_drawing = analyze_drawing(DrawingInfo['BaseImagePath'], DrawingInfo['Image_path'])
    
    return DrawingScore(
        Player_name=DrawingInfo['Player_name'],
        Player_age=DrawingInfo['Player_age'],
        ArtCategory=category,
        ArtDetailScore=detail_score,
        BaseImagePath=DrawingInfo['BaseImagePath'],
        AnalyzingImagePath=DrawingInfo['Image_path'],
        ArtScore=score,
        NameOfDrawing=name_of_drawing
    )

if __name__ == "__main__":
    result = Player_playing()
    print("\nResult:")
    print(f"Player Name: {result.Player_name}")
    print(f"Player Age: {result.Player_age}")
    print(f"Name of Drawing: {result.NameOfDrawing}")
    print(f"Art Category: {result.ArtCategory}")
    print(f"Number of Details: {result.ArtDetailScore}")
    print(f"Final Score: {result.ArtScore}")
    
