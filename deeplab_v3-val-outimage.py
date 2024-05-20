import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
print("PyTorch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=False)
model.eval()
model.to(device)

preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

input_dir = "DataSets/VOC2012/JPEGImages"
output_dir = "miou_outVOC/detection-results"

VOCdevkit_path = 'DataSets'
image_ids = open(os.path.join(VOCdevkit_path, "VOC2012/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
gt_dir = os.path.join(VOCdevkit_path, "VOC2012/SegmentationClass/")
miou_out_path = "miou_outVOC"
pred_dir = os.path.join(miou_out_path, 'detection-results')

os.makedirs(output_dir, exist_ok=True)

for image_id in tqdm(image_ids):
    image_path = os.path.join(VOCdevkit_path, "VOC2012/JPEGImages/" + image_id + ".jpg")
    input_image = Image.open(os.path.join(image_path)).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    output_predictions = output.argmax(0)

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    segmented_image = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    # segmented_image.putpalette(colors)

    output_filename = os.path.join(pred_dir, image_id + ".png")
    segmented_image.save(output_filename)
    print("finish segmentation picture:",output_filename)

print("Segmentation complete. Segmented images saved in '{}'.".format(output_dir))
