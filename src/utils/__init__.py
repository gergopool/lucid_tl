from src.utils.image import imagenet_preprocess, robi_custom_preprocess
from src.utils.config import get_config
from src.utils.vis import first_look_on_imgs

def get_preprocess_func(str_form):
    if str_form == 'imagenet':
        return imagenet_preprocess
    elif str_form == 'robi':
        return robi_custom_preprocess
    else:
        raise ValueError(
            'Preprocessing function does not exist ({})'.format(str_form))
