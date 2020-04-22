import imageio
images = []

def convert_to_gif(num):
    for i in range(num):
        images.append(imageio.imread('{}.jpg'.format(i)))
    imageio.mimsave('animation.gif', images)