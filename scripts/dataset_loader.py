#src/data/dataset_loader.py
"""
Folder layout:
data/face_mask/with_mask/
data/face_mask/without_mask/

tf.data-pipline with rescale+augment 

"""
import tensorflow as tf 
from pathlib import Path 
from src.utils.logger import logger 
from src.utils.exceptions import wrap_error

@wrap_error
def load_dataset(root:str,img_size:int,batch:int):
    root=Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found -place image ")
    
    # augmnetation layer 
    aug=tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("Horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name='augment',
    )

    train_ds=tf.keras.preprocessing.image_dataset_from_directory(
        root,
        labels='inferred',
        label_mode="categorical",
        validation_split=0.2,
        subset='training',
        seeed=42,
        image_size=(img_size,img_size),
        batch_size=batch,

    ).map(lambda x , y:(aug(x)/255.0,y))

    val_ds=tf.keras.preprocessing.image_dataset_from_directory(
        root,
        labels='inferred',
        label_mode="categorical",
        validation_split=0.2,
        subset='validation',
        seeed=42,
        image_size=(img_size,img_size),
        batch_size=batch,

    ).map(lambda x , y:(aug(x)/255.0,y))

    AUTOTUNE=tf.data.AUTOTUNE
    train_ds=train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds=val_ds.cache().prefetch(AUTOTUNE)

    logger.info("📦 Dataset loaded with augmentation")
    return train_ds,val_ds