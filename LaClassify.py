import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def mysoftmax(lst):
    result = []
    for array in lst:
        tensor = torch.from_numpy(array)  # 将NumPy数组转换为PyTorch张量
        softmax_tensor = torch.softmax(tensor, dim=1)  # 应用torch.softmax，dim=1表示对每行进行softmax
        softmax_array = softmax_tensor.numpy()  # 将PyTorch张量转换回NumPy数组
        result.append(softmax_array)
    return result

def predict_image(image, models):


    image = transform(image).unsqueeze(0).to(device)

    predictions = []
    for model in models:
        with torch.no_grad():
            outputs = model(image)
            outputs = torch.softmax(outputs, dim=1)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions)

    if len(predictions) == 1:
        predictions = predictions[0]
    else:
        predictions = predictions[0] * 0.06 + predictions[1] * 0.27 + predictions[2] * 0.67
    return predictions

def getModelNum(kind):

    model_num = [0,1,2]

    if kind < 3:
        model_num = [model_num[kind]]
    elif kind > 3:
        model_num = [3]

    return model_num


def predict_pic(image, models):

    predictions = predict_image(image, models)
    return predictions
