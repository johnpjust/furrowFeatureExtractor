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

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def load_dataset(args, data_len):
    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    data_split_ind = np.random.permutation(data_len)
    train_ind = data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]
    val_ind = data_split_ind[int((1 - 2 * args.p_val) * len(data_split_ind)):int((1 - args.p_val) * len(data_split_ind))]
    test_ind = data_split_ind[int((1 - args.p_val) * len(data_split_ind)):]

    return train_ind, val_ind, test_ind

def create_model(args):

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    actfun = tf.nn.elu

    inputs = tf.keras.Input(shape=args.img_size, name='img')  ## (108, 192, 3)
    x = layers.Conv2D(32, 7, activation=actfun, strides=3)(inputs)
    block_output = layers.Conv2D(32,3,strides=2, activation=None)(x)
    # block_output = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(64, 1, activation=actfun)(x)
    block_output = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(64, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(64, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(64, 1, activation=actfun)(x)
    # block_output = layers.MaxPooling2D(2, strides=2)(x)

    # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # x = layers.Conv2D(32, 3, activation=None)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4, activation=actfun)(x)
    quality = tf.nn.tanh(layers.Dense(1)(x))
    model = tf.keras.Model(inputs, quality, name='toy_resnet')
    model.summary()

    return model

def train(model, optimizer, scheduler, imgs, quality_y, train_ind, val_ind, test_ind, args):

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for ind in batch(np.random.permutation(train_ind), args.batch_dim):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.math.squared_difference(quality_y[ind,:], model(imgs[ind,:,:,:], training=True)))
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            globalstep = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            tf.summary.scalar('loss/train', loss, globalstep)

        ## potentially update batch norm variables manually
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        validation_loss = []
        for ind in batch(val_ind, 2*args.batch_dim):
            loss = tf.reduce_mean(tf.math.squared_difference(quality_y[ind,:], model(imgs[ind,:,:,:], training=False))).numpy()
            validation_loss.append(loss)
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        for ind in batch(test_ind, 2*args.batch_dim):
            loss = tf.reduce_mean(tf.math.squared_difference(quality_y[ind,:], model(imgs[ind,:,:,:], training=False))).numpy()
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
args.early_stopping = 10
args.manualSeed = None
args.manualSeedw = None
args.prefetch_size = 2  # data pipeline prefetch buffer size
args.parallel = 8  # data pipeline parallel processes
args.preserve_aspect_ratio = True;  ##when resizing
args.p_val = 0.2


args.path = os.path.join(args.tensorboard, 'furrowfeat_{}'.format(str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

data = np.load(r'Z:\current\Projects\Deere\Seeding\2019\Data\SeedFurrowCamera\Extracted Images\extracted_trench_quality_signals\feat_ext_preproc_data.npy', allow_pickle=True)
imgs = np.array([x for x in data[:,0]]).astype(np.float32)/np.float32(255)

quality_y = np.array([x for x in data[:,1]]).astype(np.float32).reshape(-1,1)
args.img_size = imgs[0].shape
data = []

print('Loading dataset..')
train_ind, val_ind, test_ind = load_dataset(args, imgs.shape[0])

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
    train(model, optimizer, scheduler, imgs, quality_y, train_ind, val_ind, test_ind, args)

# ###################### inference #################################
    embeds = tf.keras.Model(model.input, model.layers[-3].output, name='embeds')

    # train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    # train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/128.0 - 1
    # test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    # test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/128.0 - 1
    # all_data = np.concatenate((train_data, test_data))

    rand_crops_embeds = []
    for x in batch(imgs, 2*args.batch_dim):
        rand_crops_embeds.extend(embeds(x))

    rand_crops_embeds = np.stack(rand_crops_embeds)

    np.savetxt(r'C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard\furrowfeat_2020-01-30-18-58-47\embeds.csv', rand_crops_embeds, delimiter=',')

    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(rand_crops_embeds)
    # distances, indices = nbrs.kneighbors(np.array([-0.35703278, -0.33590597, -0.8081483 , -0.01309389]).reshape(-1,4))  ## rand_crops_embeds[30212,:] [-0.84653145, -0.14351833, -0.8278878 , -0.7618342 ]
    distances, indices = nbrs.kneighbors(np.array([-0.24539942, -0.5202056 , -0.61923814, -0.25972468]).reshape(-1, 4))
    plt.figure();plt.imshow(np.uint8(imgs[indices[0][5]]*255))

# if __name__ == '__main__':
#     main()

#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard
## http://localhost:6006/