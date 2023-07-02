import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from telegram.ext import Updater, MessageHandler, Filters

def find_similar_image(user_image_path, piccher_dir):
    # Load the user image
    user_image = cv2.imread(user_image_path)
    user_image = cv2.cvtColor(user_image, cv2.COLOR_BGR2RGB)

    # Preprocess the user image
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    user_image = preprocess(user_image)
    user_image = torch.unsqueeze(user_image, 0)

    # Load the pre-trained model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Get the output tensor for the user image
    with torch.no_grad():
        user_output = model(user_image)

    # Iterate over the images in the piccher_dir directory
    similar_image_path = None
    min_distance = float('inf')
    for filename in os.listdir(piccher_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(piccher_dir, filename)

            # Load and preprocess the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess(image)
            image = torch.unsqueeze(image, 0)

            # Get the output tensor for the image
            with torch.no_grad():
                output = model(image)

            # Calculate the distance between the user image output and the image output
            distance = torch.dist(output, user_output)

            # Update the similar image
            if distance < min_distance:
                similar_image_path = image_path
                min_distance = distance

    return similar_image_path


def handle_image(bot, update):
    # Get the file ID of the received image
    file_id = update.message.photo[-1].file_id

    # Download the image file
    new_file = bot.get_file(file_id)
    new_file.download('user_image.jpg')

    # Find similar image
    similar_image_path = find_similar_image('user_image.jpg', 'piccher_dir')

    if similar_image_path:
        # Send the similar image to the user
        bot.send_photo(update.message.chat_id, photo=open(similar_image_path, 'rb'))
    else:
        # Send a message if no similar image found
        bot.send_message(update.message.chat_id, text="No similar image found.")


# Set up the Telegram Bot
updater = Updater('6138612511:AAHtpceIAGCSZuUH8NGqPBarygmi6Uw0qJE')
dispatcher = updater.dispatcher

# Register the image handler
image_handler = MessageHandler(Filters.photo, handle_image)
dispatcher.add_handler(image_handler)

# Start the bot
updater.start_polling()
