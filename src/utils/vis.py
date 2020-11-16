from matplotlib import pyplot as plt

def first_look_on_imgs(imgs, rows=4, cols=8):
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        ax.imshow(img)
    plt.show()