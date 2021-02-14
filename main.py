import logging
import os
import random
import string

import numpy as np
import torch
import torchvision
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.neighbors import KNeighborsTransformer
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from ae_model import Autoencoder
from get_dataset import load_images

logger = logging.getLogger(__name__)

ae_model = None
knn = None
dataset = None

number_of_similar_images = 5
to_tensor = torchvision.transforms.ToTensor()


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text(
        "Hi! This is a similar face finder. Please upload face photo and I'll return you several similar faces.")


def help_command(update: Update) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text("Please upload face photo and I'll return you several similar faces.")


def handle_photo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    logging.info("Photo received")
    file_id = update.message.photo[-1].file_id

    image_file = context.bot.getFile(file_id)
    image_raw = imread(image_file.file_path)
    image_raw = resize(image_raw, [48, 48])
    image = to_tensor(image_raw).float()
    image = image.unsqueeze(dim=0)
    latent = ae_model.encode(image)
    latent = latent.detach().cpu()

    _, (neighbor_indexes,) = knn.kneighbors(latent, n_neighbors=number_of_similar_images)

    similar_images = []
    for i in range(number_of_similar_images):
        neighbour_index = neighbor_indexes[i]
        neighbour_image = dataset[neighbour_index]
        similar_images.append(neighbour_image)

    merged_image = np.concatenate(similar_images, axis=1)

    temp_file_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".jpeg"
    imsave(temp_file_name, merged_image)
    temp_file_content = open(temp_file_name, 'rb')
    update.message.reply_photo(temp_file_content)

    os.remove(temp_file_name)


def main():
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    load_model()
    load_dataset()
    train_knn()

    logging.info("Preparation done, starting Telegram bot")

    updater = Updater("1548635155:AAHcArsfr_KDFW1Fj00r5elM61ARtSqvYog")
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.photo & ~Filters.command, handle_photo))

    updater.start_polling()
    updater.idle()


def load_dataset():
    logging.info("Loading dataset")
    global dataset
    dataset = load_images()


def load_model():
    logging.info("Loading AE model")
    global ae_model
    ae_model = Autoencoder((3, 48, 48), 256)
    ae_model.load_state_dict(torch.load("face_ae.pth"))
    ae_model.eval()


def train_knn():
    logging.info("Training KNN")
    latent_codes = []

    to_tensor = torchvision.transforms.ToTensor()

    for image in dataset:
        image = to_tensor(image).float()
        image = image.unsqueeze(dim=0)
        latent = ae_model.encode(image)
        latent = latent.detach().cpu()

        latent_codes.append(latent)

    latent_codes = np.vstack(latent_codes)

    global knn
    knn = KNeighborsTransformer().fit(latent_codes)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
