from src.utils.image import imagenet_preprocess, robi_custom_preprocess, tensorflow_preporcess
from src.utils.config import get_config
from src.utils.vis import first_look_on_imgs
from torchvision import transforms

def get_preprocess_func(str_form):
    if str_form == 'imagenet':
        return imagenet_preprocess
    elif str_form == 'robi':
        return robi_custom_preprocess
    elif str_form == 'tf':
        return tensorflow_preporcess
    else:
        raise ValueError(
            'Preprocessing function does not exist ({})'.format(str_form))


shift = 117
mean = shift/127.5-0.5
std = 1/256.

celeba_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.Pad(10, padding_mode='edge'),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((mean, mean, mean), (std, std, std))
])

celeba_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((mean, mean, mean), (std, std, std))
])

