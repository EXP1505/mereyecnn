from dataloader import *
from TRAINING_CONFIG import *
import torchvision
from PIL import Image
import torch


def eval(test_model):
    
  test_dataloader = data_prep(testing=True)

  test_model.eval()

  if not os.path.exists(output_images_path):
    os.makedirs(output_images_path, exist_ok=True)

  for i, (img, _, name) in enumerate(test_dataloader):
    with torch.no_grad():
      if img.size()[1] == 4:    #if alpha channel exists in test images, remove alpha channel
        img = img[:, :3, :, :]
      img = img.to(device)
      generate_img = test_model(img)
      # Clamp values to [0, 1] range and ensure proper format
      generate_img = torch.clamp(generate_img, 0, 1)
      
      # Convert tensor to PIL Image and save
      generate_img_np = generate_img.squeeze(0).cpu().numpy()
      generate_img_np = (generate_img_np * 255).astype('uint8')
      generate_img_np = generate_img_np.transpose(1, 2, 0)
      pil_img = Image.fromarray(generate_img_np)
      pil_img.save(output_images_path + name[0])

  print("Evaluation of Given Test Images Completed!")

def run_testing():
  model_test = torch.load(test_model_path, weights_only=False, map_location=device)
  eval(model_test)


### START TESTING ###
#run_testing()
