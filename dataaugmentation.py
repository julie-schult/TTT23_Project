from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import cv2
import numpy as np

def augmentation(img):
    image = img.reshape((1,) + img.shape)
    
    datagen = ImageDataGenerator(
        rotation_range=40,  # Randomly rotate images by up to 40 degrees
        zoom_range=0.3,  # Zoom in/out randomly by up to 30%
        horizontal_flip=True,  # Flip images horizontally
        brightness_range=[0.5, 1.5]  # Adjust brightness randomly in the range [0.5, 1.5]
    )
    
    # Generate augmented images
    augmented_images = []
    for batch in datagen.flow(image, batch_size=1):
        augmented_images.append(batch[0])
        if len(augmented_images) >= 2:  # Generate 2 augmented images
            break
        
    return augmented_images


def saveImagesAndCreateDF(csvpath):
    paths = []
    bodys = []
    heads = []
    lungs = []

    root = 'data'

    df = pd.read_csv(csvpath, delimiter=';')
    #print(df)

    for index, row in df.iterrows():
        path = row[0]
        body = row[1]
        head = row[2]
        lung = row[3]
        
        print(path, body, head, lung)
        
        name = os.path.basename(path).split('.')[0]
        
        image = cv2.imread(path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Adding the original image to df, and save to folder
        new_name = name + '_original' + ".jpg"
        subpath = os.path.join(root, 'original')
        subsubpath = os.path.join(subpath, new_name)
        cv2.imwrite(subsubpath, image)
        paths.append(subsubpath)
        bodys.append(body)
        heads.append(head)
        lungs.append(lung)
        
        # Adding augmentated images to df, and save to folder
        augmented_images = augmentation(image)
        for i, aug_img in enumerate(augmented_images):
            new_name = name + '_augmented' + str(i+1) + ".jpg"
            subpath = os.path.join(root, 'augmented')
            subsubpath = os.path.join(subpath, new_name)

            img_aug = aug_img.astype(np.uint8)
            cv2.imwrite(subsubpath, img_aug)
            paths.append(subsubpath)
            bodys.append(body)
            heads.append(head)
            lungs.append(lung)

    data = {
        'path' : paths,
        'body' : bodys,
        'head' : heads,
        'lung' : lungs
    }
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)