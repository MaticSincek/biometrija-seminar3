import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import shutil
import os

lines = []

with open("awe-translation.csv") as file:
    lines = file.readlines()

for line in lines:
    split = line.split(",")
    c = int(split[0])-1
    filename = split[1]
    tt = split[2][0:3]

    print(c)
    print(filename)
    print(tt)

    if tt == "tes":
        f = "./awe/" + filename
        t = "./testing/" + str(c) + "_" + filename.split("/")[1]
        shutil.copyfile(f , t)
        os.remove(f)

for dirname in  os.listdir("./awe"):
    try:
        os.remove("./awe/" + dirname + "/annotations.json")
    except:
        print("Exception thrown. x does not exist.")

image_size = (100, 100)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "awe",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "awe",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=101)
keras.utils.plot_model(model, show_shapes=True)

#training
epochs = 40

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

#testing
t = 0
f = 0

for dirname in os.listdir("./testing"):
  c = dirname.split("_")[0]

  img = keras.preprocessing.image.load_img(
      "./testing/" + dirname, target_size=image_size
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)  # Create batch axis

  predictions = model.predict(img_array)

  max = 0
  maxi = 0
  for i in range(100):
    if predictions[0][i] > max:
      max = predictions[0][i]
      maxi = i

  #print(str(maxi) + " - " + c)
  if str(maxi) == c:
    t += 1
  else:
    f += 1

print(t / (t+f))

#natančnost na 60 epoch-ih: 17%