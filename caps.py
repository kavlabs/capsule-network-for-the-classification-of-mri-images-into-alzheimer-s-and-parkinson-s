import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')

train_data_dir = '/content/drive/My Drive/Comparison paper folder with 1200 MRI'
val_split = 0.3
totalpics = 4800

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.show()
    plt.savefig('acc_vs_epochs.png')

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        a1 = 0
        a2 = 0
        a3 = 0
        a4 = 0
        for u in range(self.test_data.samples // batch_size):
            x, y = self.test_data[u]
            pred = self.model.predict(x)
            true = y
            prediction = np.argmax(pred,axis=1)
            label = np.argmax(true,axis=1)
            acc1 = 0
            acc2 = 0
            acc3 = 0
            acc4 = 0
            tar1 = label[label==0]
            tar2 = label[label==1]
            tar3 = label[label==2]
            tar4 = label[label==3]
            size_of_AD = len(tar1)
            size_of_nAD = len(tar2)
            size_of_nPD = len(tar3)
            size_of_PD = len(tar4)
            for i in range(len(label)):
                if label[i]==0:
                    if prediction[i]==0:
                        acc1 += 1/size_of_AD
            a1+=acc1
            for i in range(len(label)):
                if label[i]==1:
                    if prediction[i]==1:
                        acc2 += 1/size_of_nAD
            a2+=acc2
            for i in range(len(label)):
                if label[i]==2:
                    if prediction[i]==2:
                        acc3 += 1/size_of_nPD
            a3+=acc3
            for i in range(len(label)):
                if label[i]==3:
                    if prediction[i]==3:
                        acc4 += 1/size_of_PD
            a4+=acc4
        print('\n AD accuracy:{}\n'.format(a1/9))
        print('\n nAD accuracy:{}\n'.format(a2/9))
        print('\n nPD accuracy:{}\n'.format(a3/9))
        print('\n PD accuracy:{}\n'.format(a4/9))

def CapsNet(input_shape, n_class, routings, batch_size):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)
    out = models.Model(inputs=x, outputs=out_caps)

    return out

def margin_loss(y_true, y_pred):
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train(model,args):
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    ##my block----------------------------------------------------------------------------------------------------#
    train_datagen = ImageDataGenerator(validation_split=val_split,
                                   rescale=1. / 255)
 
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        shuffle=False,
        target_size=(img_height, img_width),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training')
     
    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        shuffle=False,
        target_size=(img_height, img_width),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='validation')

    k = TestCallback(validation_generator)

    model.fit(
        train_generator,
        steps_per_epoch=(totalpics*(1-val_split)) // args.batch_size,
        epochs=args.epochs,
        callbacks=[log, checkpoint, lr_decay, k],
        validation_data=validation_generator,
        validation_steps=(totalpics*(val_split)) // args.batch_size)
    ##------------------------------------------------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model

if __name__ == "__main__":
    import os
    import argparse
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    img_width, img_height = 64, 64
    ch = 3

    # define model
    model = CapsNet(input_shape=(img_width,img_height,ch),
                                                  n_class=4,
                                                  routings=args.routings,
                                                  batch_size=args.batch_size)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
        )

    # train or test
    if args.weights is not None:
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, args=args)
    else:
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, args)
        test(model=eval_model, args=args)