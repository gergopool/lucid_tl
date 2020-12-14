
def load_layer_info(path='archive/torch_layer_info.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        layer_name, n_channels = line.split(' ')
        n_channels = int(n_channels)
        data.append([layer_name, n_channels])
    return data