import celebalucid as cl
from tqdm import tqdm
import os
import cv2
import torch
import sys

if './' not in sys.path:
    sys.path.append('./')

save_dir_root = 'results/lucid_vis_some/'
model_path = sys.argv[1]
model_name = sys.argv[2]
batch_norm = bool(int(sys.argv[3]))

model = cl.load_model('imagenet')
model.bn = batch_norm
model.load_state_dict(torch.load(model_path), strict=False)
model.weights = model._load_weights()

for layer, _ in tqdm(model.layer_info):
    save_dir = os.path.join(save_dir_root, model_name, layer)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(10):
        vis_layer = layer + ':' + str(i)
        img = model.lucid(vis_layer, progress=False)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(save_dir, str(i)+'.jpg')
        cv2.imwrite(save_path, img)