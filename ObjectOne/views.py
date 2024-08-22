from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Create your views here.
def home(request):
    return render(request,'index.html')

def upload_image(request):
    if request.method=='POST':
        image = request.FILES['image']
        file_name = default_storage.save(image.name, ContentFile(image.read()))
        file_path = default_storage.path(file_name)

        # Process image with OpenCV
        plate_number = detect_license_plate(file_path)

        return render(request, 'result.html', {
            'image_url': file_path,
            'plate_number': plate_number,
        })
    return render(request,'index.html')

def detect_license_plate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    plates = plate_cascade.detectMultiScale(thresh, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(plates) > 0:
        x, y, w, h = plates[0]
        plate_image = image[y:y+h, x:x+w]
        plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        # Apply additional preprocessing to plate image
        _, plate_image_thresh = cv2.threshold(plate_image_gray, 127, 255, cv2.THRESH_BINARY)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        plate_number = pytesseract.image_to_string(plate_image_thresh, config='--psm 8')
        plate_number = plate_number.strip()
    else:
        plate_number = "No plate detected"

    return plate_number
