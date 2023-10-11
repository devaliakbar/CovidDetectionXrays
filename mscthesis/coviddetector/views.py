from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render


def upload_file(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage(location='coviddetector/upload')
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)
        result = predict_covid(file_path)
        return render(request, 'result.html', {'result': 'Covid Detected!' if result else 'Covid Not Detected!'})
    else:
        return render(request, 'upload.html')


def get_image_arr(path):
    img_path = path
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array


def predict_covid(image_path):
    scratch = load_model('Output/ScratchModel.h5')
    resnet = load_model('Output/ResnetModel.h5')
    xception = load_model('Output/XceptionModel.h5')

    image = get_image_arr(image_path)

    scratch_result = False if scratch.predict(image) >= 0.5 else True
    resnet_result = False if resnet.predict(image) >= 0.5 else True
    xception_result = False if xception.predict(image) >= 0.5 else True

    return sum([scratch_result, resnet_result, xception_result]) >= 2
