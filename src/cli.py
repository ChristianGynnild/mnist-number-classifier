import click
import dataset as _dataset
import model
import constants

@click.group()
def cli():
    pass

@cli.command()
def train():
    model.train()

@cli.command()
@click.argument('image_path')
def predict(image_path):
    model.predict(image_path)
    
    
@cli.group()
def test():
    pass

@test.command()
def dataset():
    print(constants.TRAINING_IMAGES_PATH)
    data = _dataset.load_images(constants.TRAINING_IMAGES_PATH)
    for x in range(28):
        for y in range(28):
            print(data[0][x][y])
    _dataset.image_to_file(data[0], "image.png")
    
