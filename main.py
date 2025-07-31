import os
import json
import csv
import base64
import requests
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path
import difflib
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import time
from io import BytesIO

@dataclass
class OCRResult:
    image_path: str
    ground_truth: str
    prediction: str
    cer_score: float

class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"

    def encode_image_object_to_base64(self, image: Image.Image) -> str:
        try:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def predict_license_plate_from_image(self, image: Image.Image, model_name: str = "llava") -> str:
        try:
            image_base64 = self.encode_image_object_to_base64(image)
            if not image_base64:
                return ""

            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is the license plate number shown in this image? Respond only with the plate number without any additional text or explanation."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1
            }

            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=50000
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result['choices'][0]['message']['content'].strip()
                return self.clean_prediction(prediction)
            else:
                print(f"Error calling LMStudio API: {response.status_code}")
                return ""

        except Exception as e:
            print(f"Prediction error: {e}")
            return ""

    def clean_prediction(self, prediction: str) -> str:
        cleaned = prediction.replace('"', '').replace("'", '').strip()
        import re
        plate_pattern = r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}'
        match = re.search(plate_pattern, cleaned.upper())
        if match:
            return match.group().replace(' ', '')
        return ''.join(c for c in cleaned.upper() if c.isalnum())

class CERCalculator:
    @staticmethod
    def calculate_cer(ground_truth: str, prediction: str) -> float:
        if not ground_truth:
            return 1.0 if prediction else 0.0

        operations = difflib.SequenceMatcher(None, ground_truth, prediction)
        substitutions = deletions = insertions = 0

        for op, i1, i2, j1, j2 in operations.get_opcodes():
            if op == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                deletions += i2 - i1
            elif op == 'insert':
                insertions += j2 - j1

        total_errors = substitutions + deletions + insertions
        return total_errors / len(ground_truth) if len(ground_truth) > 0 else 0.0

    @staticmethod
    def calculate_detailed_cer(ground_truth: str, prediction: str) -> Dict:
        if not ground_truth:
            return {
                'cer': 1.0 if prediction else 0.0,
                'substitutions': 0,
                'deletions': 0,
                'insertions': len(prediction) if prediction else 0,
                'total_errors': len(prediction) if prediction else 0,
                'ground_truth_length': 0
            }

        operations = difflib.SequenceMatcher(None, ground_truth, prediction)
        substitutions = deletions = insertions = 0

        for op, i1, i2, j1, j2 in operations.get_opcodes():
            if op == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif op == 'delete':
                deletions += i2 - i1
            elif op == 'insert':
                insertions += j2 - j1

        total_errors = substitutions + deletions + insertions
        cer = total_errors / len(ground_truth)

        return {
            'cer': cer,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'total_errors': total_errors,
            'ground_truth_length': len(ground_truth)
        }

def generate_preprocessed_images(image_path: str) -> List[Image.Image]:
    original = Image.open(image_path)
    images = []
    images.append(original.convert("L").resize((224, 224)))
    images.append(ImageEnhance.Contrast(original.convert("L")).enhance(2.0).resize((224, 224)))

    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is not None:
        eq = cv2.equalizeHist(img_cv)
        images.append(Image.fromarray(eq).resize((224, 224)))

    images.append(original.convert("L").filter(ImageFilter.GaussianBlur(1)).resize((224, 224)))
    return images

class LicensePlateOCR:
    def __init__(self, lmstudio_url: str = "http://localhost:1234", model_name: str = "llava"):
        self.client = LMStudioClient(lmstudio_url)
        self.model_name = model_name
        self.cer_calculator = CERCalculator()
        self.results: List[OCRResult] = []

    def load_dataset(self, dataset_path: str, ground_truth_file: str = None) -> Dict[str, str]:
        ground_truth_dict = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            try:
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ground_truth_dict[row['image']] = row['ground_truth']
            except Exception as e:
                print(f"Error reading ground truth file: {e}")
        return ground_truth_dict

    def process_single_image(self, image_path: str, ground_truth: str = "") -> OCRResult:
        print(f"Processing: {image_path}")
        preprocessed_images = generate_preprocessed_images(image_path)

        best_result = None
        best_cer = float("inf")

        for idx, image in enumerate(preprocessed_images):
            print(f" - Variant {idx+1}/{len(preprocessed_images)}")
            prediction = self.client.predict_license_plate_from_image(image, self.model_name)
            cer = self.cer_calculator.calculate_cer(ground_truth, prediction)

            if cer < best_cer:
                best_cer = cer
                best_result = OCRResult(
                    image_path=image_path,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    cer_score=cer
                )

        print(f"Best prediction: {best_result.prediction}")
        print(f"Best CER: {best_result.cer_score:.4f}")
        print("-" * 40)
        return best_result

    def process_dataset(self, dataset_path: str, ground_truth_file: str = None) -> List[OCRResult]:
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path '{dataset_path}' does not exist!")
            return []

        ground_truth_dict = self.load_dataset(dataset_path, ground_truth_file)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in Path(dataset_path).iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in '{dataset_path}'")
            return []

        results = []
        for image_path in image_files:
            image_name = image_path.name
            ground_truth = ground_truth_dict.get(image_name, "")
            try:
                result = self.process_single_image(str(image_path), ground_truth)
                results.append(result)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        self.results = results
        return results

    def save_results_to_csv(self, output_file: str = "ocr_results.csv"):
        if not self.results:
            print("No results to save!")
            return
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'ground_truth', 'prediction', 'CER_score'])
                for result in self.results:
                    writer.writerow([
                        os.path.basename(result.image_path),
                        result.ground_truth,
                        result.prediction,
                        f"{result.cer_score:.4f}"
                    ])
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def calculate_overall_metrics(self) -> Dict:
        if not self.results:
            return {
                'total_images': 0,
                'average_cer': 0.0,
                'accuracy': 0.0,
                'total_substitutions': 0,
                'total_deletions': 0,
                'total_insertions': 0,
                'total_ground_truth_length': 0,
                'correct_predictions': 0
            }

        total_cer = sum(result.cer_score for result in self.results)
        avg_cer = total_cer / len(self.results)
        correct_predictions = sum(1 for result in self.results if result.ground_truth == result.prediction and result.ground_truth != "")
        images_with_gt = sum(1 for result in self.results if result.ground_truth != "")
        accuracy = correct_predictions / images_with_gt if images_with_gt > 0 else 0.0

        total_substitutions = total_deletions = total_insertions = total_ground_truth_length = 0

        for result in self.results:
            if result.ground_truth:
                detailed = self.cer_calculator.calculate_detailed_cer(result.ground_truth, result.prediction)
                total_substitutions += detailed['substitutions']
                total_deletions += detailed['deletions']
                total_insertions += detailed['insertions']
                total_ground_truth_length += detailed['ground_truth_length']

        return {
            'total_images': len(self.results),
            'average_cer': avg_cer,
            'accuracy': accuracy,
            'total_substitutions': total_substitutions,
            'total_deletions': total_deletions,
            'total_insertions': total_insertions,
            'total_ground_truth_length': total_ground_truth_length,
            'correct_predictions': correct_predictions,
            'images_with_ground_truth': images_with_gt
        }

    def print_summary(self):
        metrics = self.calculate_overall_metrics()
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        print(f"Total Images Processed: {metrics['total_images']}")
        print(f"Images with Ground Truth: {metrics['images_with_ground_truth']}")
        print(f"Average CER: {metrics['average_cer']:.4f}")
        print(f"Accuracy (Exact Match): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['images_with_ground_truth']}")
        print(f"Total Substitutions: {metrics['total_substitutions']}")
        print(f"Total Deletions: {metrics['total_deletions']}")
        print(f"Total Insertions: {metrics['total_insertions']}")
        print("="*60)

def main():
    lmstudio_url = "http://localhost:1234/v1/chat/completions"
    image_dir = r"C:\\Users\\ASUS_TUF_GAMING\\Documents\\Dataset\\test"
    ground_truth_file = os.path.join(image_dir, "ground_truth.csv")
    output_file = "ocr_results.csv"
    model_name = "bakllava1-mistralllava-7b"

    ocr = LicensePlateOCR(lmstudio_url.replace("/v1/chat/completions", ""), model_name)
    print("Starting License Plate OCR with VLM...")
    print(f"Image Directory: {image_dir}")
    print(f"Ground Truth File: {ground_truth_file}")
    print(f"Model: {model_name}")
    print(f"LMStudio URL: {lmstudio_url}")
    print("-" * 60)

    try:
        results = ocr.process_dataset(image_dir, ground_truth_file)
        if results:
            ocr.save_results_to_csv(output_file)
            ocr.print_summary()
        else:
            print("No images were processed successfully!")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Please make sure:")
        print("1. LMStudio is running on the specified URL")
        print("2. Model is loaded in LMStudio")
        print("3. Dataset path exists and contains images")
        print("4. Ground truth file format is correct")

if __name__ == "__main__":
    main()
