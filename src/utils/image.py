def imagenet_preprocess(img):
    # Works both for single img or batch of images
    return (img/255.-0.5)*2 

def robi_custom_preprocess(img):
    return img/255.