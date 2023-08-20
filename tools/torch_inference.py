from torchvision import models
import cv2
import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import time 

def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        ToTensor(),
        Resize(224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(input_img)
    batch_data = torch.unsqueeze(input_data, 0)
    return batch_data

def postprocess(output_data, verbose):
    # get class names
    with open("data/imagenet-classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 50:
        class_idx = indices[0][i]
        if verbose:
             print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1
        
        
print("=================== STARTING PYTORCH INFERENCE===============================")

input = preprocess_image("data/hotdog.jpg").cuda()
model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
model.eval()
model.cuda()
with torch.no_grad():
    output = model(input)
postprocess(output, True)

# save network output
op = output.clone().cpu().numpy()
file_path_op = 'torch_stuff/output.txt'
np.savetxt(file_path_op, op, fmt='%f', delimiter=',')  # Use fmt='%d' for integers
print("Saved Pytorch output in torch_stuff/output.txt")

# save average latency for 10 iterations
num_iterations = 10
total_latency = 0

with torch.no_grad():
    for _ in range(num_iterations):
        input = preprocess_image("data/hotdog.jpg").cuda()
        start_time = time.time()
        output = model(input)
        end_time = time.time()
        postprocess(output, False)
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        total_latency += latency

average_latency = total_latency / num_iterations
    
print(f"Average Latency for {num_iterations} iterations: {average_latency:.2f} ms")

# Write the average latency to a text file
with open("torch_stuff/latency.txt", "w") as f:
    f.write(f"{average_latency:.2f}")
    
print("Saved Pytorch output in torch_stuff/output.txt")

print("=======================================================================")
