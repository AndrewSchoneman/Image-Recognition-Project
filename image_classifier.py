# -*- coding: utf-8 -*-
"""

@author: Andy
"""

from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras import backend as K
from keras.models import load_model

# specify the model type for initial training
# choices are VGG 19 or VGG 16, VGG 19 is enormous 
# and can be difficult to fit in memory for even a 
# decent GPU. Can be run on CPU but very slow and 
# will eat up a lot of memory
model_type = "VGG 16"

# specify whether you are reloading and traing
reload = False

# The dimensions of our images for Zappos 50k they are
# 136 x 102
img_width, img_height = 136, 102



''' Image modification settings '''
# Don't change the rescale settings keras documentation
# specifies using this exact recale
rescale=1. / 255
shear_range=0.2
zoom_range=0.2
rotation_range = 0
horizontal_flip=True
vertical_flip= True

''' Training settings '''
steps_per_epoch = 150
epochs = 120
batch_size = 125
dropout = 0.55
regularization = False 
regularization_setting = 0.01

''' Save paths '''
# Save all checkpoints doesn't save every checkpoint only ones that have improved
# from the previous high. 
save_all_checkpoints = False
checkpoints = "checkpoints/checkpoints/weights-improvement.hd5"
model_location = "checkpoints/Checkpoints/modelComp.h5"

#############################################



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

if not reload:
    
    # Specify the directory for the training and validation images
    train_data_dir = 'Superclasses/train'
    validation_data_dir = 'Superclasses/validation'
    
    
    if model_type == "VGG 16":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        if regularization:
            model.add(Dense(64 , kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1(0.01)))
        else:
            model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(4))
        model.add(Activation('softmax'))
    
    if model_type == "VGG 19":
        model = Sequential()
        
        ''' Architecture based off the VGG19 image recognition model
            It is not recommened that this architecture be run on a CPU '''
        
        # Block 1
        model.add(ZeroPadding2D((1,1),input_shape=input_shape))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        # Block 2
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        # Block 3
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        
        # Block 4
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        
        # Block 5
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        # Output 
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))
    
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=rescale,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip= vertical_flip,
        fill_mode='nearest')
    
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    
    if save_all_checkpoints:
        filepath="checkpoints/reload and train/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    else:
        filepath = checkpoints
        
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    model.fit_generator(
        train_generator,
        steps_per_epoch= steps_per_epoch,
        epochs=epochs,
        callbacks = callbacks_list,
        validation_data=validation_generator,
        validation_steps= 40)
    model.save(model_location)
    print("Saved model to disk")


if reload:
    ''' This is the section for reloading and training there are three methods to choose
        from one is much deeper than the other and is more of a step toward VGG 19 structure
        The final one is just to reload the pretrained model and weights of the superclasses and 
        just now make predicitons based on the new subcategories'''
   
    
    model_type = "EXP 1"
    train_data_dir = 'Subclasses/train'
    validation_data_dir = 'Subclasses/validation'



    if model_type == "EXP 1":
        old_model = load_model('checkpoints/checkpoints2/model.h5')
        all_layers = old_model.layers
        for i in range(len(all_layers)):
            all_layers[i].trainable = False
        old_model.summary()
        del old_model.layers[-6:]
            
        
        
        model = Sequential()
        for i in range(len(old_model.layers)):
            model.add(old_model.layers[i])

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))    
               
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        
        
        model.add(Flatten())
        model.add(Dense(512 ))
        model.add(Activation('relu'))
        model.add(Dropout(0.6))
        model.add(Dense(11))
        model.add(Activation('softmax'))
    
       
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    
    elif model_type == "EXP 2":
        old_model = load_model('checkpoints/checkpoints2/model.h5')
        all_layers = old_model.layers
        for i in range(len(all_layers)):
            all_layers[i].trainable = False
        del old_model.layers[-6:]
            
        
        
        model = Sequential()
        for i in range(len(old_model.layers)):
            model.add(old_model.layers[i])
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))    
               
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))
        
        model.add(Flatten())
        model.add(Dense(4096 ))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Dense(11))
        model.add(Activation('softmax'))

   
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy']) 
    
    
    else:
        model = Sequential()
        old_model = load_model('checkpoints/checkpoints2/model.h5')
        del old_model.layers[-2:]
        
        for i in range(len(old_model.layers)):
            model.add(old_model.layers[i])
        
        model.add(Dense(11))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
        
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=rescale,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip= vertical_flip,
        fill_mode='nearest')
    
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    if save_all_checkpoints:
        filepath="checkpoints/reload and train/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    else:
        filepath = checkpoints
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    model.fit_generator(
        train_generator,
        steps_per_epoch= steps_per_epoch,
        epochs=epochs,
        callbacks = callbacks_list,
        validation_data=validation_generator,
        validation_steps= 40)


    model.save(model_location)


    print("Saved model to disk")