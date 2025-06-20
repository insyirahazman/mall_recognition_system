from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np
import joblib

# Load your trained models and PCA (load once outside the view for efficiency)
pca = joblib.load('pca_model.pkl')
knn = joblib.load('knn_model.pkl')
svm = joblib.load('svm_model.pkl')

def predict_mall(request):
    prediction = None
    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        file_path = default_storage.save('tmp/' + image_file.name, image_file)
        img = Image.open(file_path).resize((128, 128)).convert('L')
        img_np = np.array(img) / 255.0
        img_np = img_np.reshape(1, -1)
        img_pca = pca.transform(img_np)
        prediction = knn.predict(img_pca)[0]  # or svm.predict(img_pca)[0]
    return render(request, 'index.html', {'prediction': prediction})