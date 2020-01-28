import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras import layers, Input, Model

import tensorflow_probability as tfp
import numpy as np
from earlystopping import *
import glob
import random
import pathlib
import scipy
import sklearn.mixture
from skimage.transform import resize

# def img_preprocessing(x_in, args):
#
#     return rand_crop, tf.squeeze(tf.matmul(tf.reshape(rand_crop, [1,-1]), args.vh))

def load_dataset(args, listing):
    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    preproc_imgs_ = preproc_imgs(listing[0])

    img_preprocessing_ = preproc_imgs_.proc_imgs

    data_split_ind = np.random.permutation(len(listing))
    train_ind = data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]
    val_ind = data_split_ind[int((1 - 2 * args.p_val) * len(data_split_ind)):int((1 - args.p_val) * len(data_split_ind))]
    test_ind = data_split_ind[int((1 - args.p_val) * len(data_split_ind)):]

    # dataset_train = tf.data.Dataset.from_tensor_slices(listing[train_ind])  # .float().to(args.device)
    # dataset_train = dataset_train.shuffle(buffer_size=len(train_ind)).map(img_preprocessing_,
    #     num_parallel_calls=args.parallel).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    # # dataset_train = dataset_train.shuffle(buffer_size=len(train)).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)
    #
    # dataset_valid = tf.data.Dataset.from_tensor_slices(listing[val_ind])  # .float().to(args.device)
    # dataset_valid = dataset_valid.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(
    #     batch_size=args.batch_dim * 2).prefetch(buffer_size=args.prefetch_size)
    # # dataset_valid = dataset_valid.batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)
    #
    # dataset_test = tf.data.Dataset.from_tensor_slices(listing[test_ind])  # .float().to(args.device)
    # dataset_test = dataset_test.map(img_preprocessing_, num_parallel_calls=args.parallel).batch(
    #     batch_size=args.batch_dim * 2).prefetch(buffer_size=args.prefetch_size)

    return dataset_train, dataset_valid, dataset_test

def create_model(args):

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    actfun = tf.nn.elu

    inputs = tf.keras.Input(shape=args.rand_box, name='img')  ## (108, 192, 3)
    x = layers.Conv2D(32, 7, activation=actfun, strides=2)(inputs)
    block_output = layers.Conv2D(32,3,strides=2, activation=None)(x)
    # block_output = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(64, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(64, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(64, 1, activation=actfun)(x)
    # x = layers.AveragePooling2D(pool_size=3, strides=2)(x)
    #
    # x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # block_output = layers.AveragePooling2D(2, strides=2)(x)

    # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # x = layers.Conv2D(32, 3, activation=None)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(3, activation=actfun)(x)
    quality = tf.nn.sigmoid(layers.Dense(1)(x))
    model = tf.keras.Model(inputs, quality, name='toy_resnet')
    model.summary()

    return model

def train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args):

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for x_mb, y_mb in data_loader_train:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.trainable_variables)
                loss = tf.reduce_mean(tf.math.squared_difference(y_mb, model(x_mb, training=True)))
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            globalstep = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            tf.summary.scalar('loss/train', loss, globalstep)

        ## potentially update batch norm variables manually
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        validation_loss = []
        for x_mb, y_mb in data_loader_val:
            loss = tf.reduce_mean(tf.math.squared_difference(y_mb, model(x_mb, training=False))).numpy()
            validation_loss.append(loss)
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        for x_mb, y_mb in data_loader_test:
            loss = tf.reduce_mean(tf.math.squared_difference(y_mb, model(x_mb, training=False))).numpy()
            test_loss.append(loss)
        test_loss = tf.reduce_mean(test_loss)

        # print("test loss:  " + str(test_loss))

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)

        #### tensorboard
        # tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/validation', validation_loss, globalstep)
        tf.summary.scalar('loss/test', test_loss, globalstep) ##tf.compat.v1.train.get_global_step()

        if stop:
            break

def load_model(args, root):
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))

class parser_:
    pass


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# tf.compat.v1.enable_eager_execution(config=config)

# tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.dataset = 'furrow'  # 'gq_ms_wheat_johnson'#'gq_ms_wheat_johnson' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
args.batch_dim = 100
args.clip_norm = 0.1
args.epochs = 5000
args.patience = 10
args.load = r''
args.save = True
args.tensorboard = r'C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard'
args.early_stopping = 50
args.manualSeed = None
args.manualSeedw = None
args.prefetch_size = 2  # data pipeline prefetch buffer size
args.parallel = 8  # data pipeline parallel processes
args.preserve_aspect_ratio = True;  ##when resizing
args.p_val = 0.2


args.path = os.path.join(args.tensorboard, 'furrowfeat_{}'.format(str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

listing = []
listing.extend(glob.glob(r'Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\West Bilsland Left Wing Log 15\*.png'))
listing.extend(glob.glob(r'Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\West Bilsland Left Wing Log 17\*.png'))
listing.extend(glob.glob(r'Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\West Bilsland Left Wing Log 19\*.png'))

print('Loading dataset..')
data_loader_train, data_loader_valid, data_loader_test = load_dataset(args, listing)

if args.save and not args.load:
    print('Creating directory experiment..')
    pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

# pathlib.Path(args.tensorboard).mkdir(parents=True, exist_ok=True)

print('Creating model..')
with tf.device(args.device):
    model = create_model(args)

## tensorboard and saving
writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
writer.set_as_default()
tf.compat.v1.train.get_or_create_global_step()

global_step = tf.compat.v1.train.get_global_step()
global_step.assign(0)

root = None
args.start_epoch = 0

print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.compat.v1.train.get_global_step())

if args.load:
    load_model(args, root)

print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

with tf.device(args.device):
    train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args)

# ###################### inference #################################
    embeds = tf.keras.Model(model.input, model.layers[-2].output, name='embeds')

    # train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    # train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/128.0 - 1
    # test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    # test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/128.0 - 1
    # all_data = np.concatenate((train_data, test_data))

    rand_crops_imgs = []
    rand_crops_embeds = []
    for _ in range(10):
        temp = [x for x in data_loader_train]
        rand_crops_imgs.extend(temp[0][0].numpy())
        rand_crops_embeds.extend(embeds(temp[0][0]))

    for _ in range(10):
        temp = [x for x in data_loader_test]
        rand_crops_imgs.extend(temp[0][0].numpy())
        rand_crops_embeds.extend(embeds(temp[0][0]))

    rand_crops_imgs = np.stack(rand_crops_imgs)
    rand_crops_embeds = np.stack(rand_crops_embeds)

# if __name__ == '__main__':
#     main()

#### tensorboard --logdir=D:\pycharm_projects\GQC_self_supervised
## http://localhost:6006/