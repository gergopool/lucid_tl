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

celeba_augment_rich = celeba_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.Pad(20, padding_mode='edge'),
    transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=10, fillcolor=0),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    lambda x: x * 255 - shift
])

celeba_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.Pad(10, padding_mode='edge'),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    lambda x: x * 255 - shift
])

celeba_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    lambda x: x * 255 - shift
])

