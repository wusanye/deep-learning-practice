#########################################
# by wu kai, 2018-07-28
#########################################


import os
from datetime import datetime
import tensorflow as tf
from alexnet_new import AlexNet
from utils import load_dataset, preprocess_data, get_batch


"""
parameter setup
"""

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

learn_rate = 0.001
lr_decay_step = 20
lr_decay_rate = 0.5
epochs_num = 80
batch_size = 32
class_num = 10
train_layer = []

display_step = 100

filewriter_path = "./tmp/tensorboard"
checkpoint_path = "./tmp/checkpoints"

"""
main part for training
"""
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

x_train, y_train, x_test, y_test, x_val, y_val = load_dataset('./cifar-10-python/', class_num)

image_shape = [32, 32, 3]
x_train, y_train = preprocess_data(x_train, y_train, image_shape)
x_val, y_val = preprocess_data(x_val, y_val, image_shape)

x = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
y = tf.placeholder(tf.float32, [batch_size, class_num])
keep_prob = tf.placeholder(tf.float32)

# model = AlexNet(x, keep_prob, class_num, train_layer)
model = AlexNet(x, class_num, keep_prob)

scores = model.scores

with tf.name_scope("cross_entropy"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y))

optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate, momentum=0.9, use_nesterov=False).minimize(loss)

with tf.name_scope("accuracy"):
    correct_preds = tf.equal(tf.argmax(scores, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

saver = tf.train.Saver()

train_batches_epoch = int(x_train.shape[0] / batch_size)
val_batches_epoch = int(x_val.shape[0] / batch_size)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print("{} Start training...".format(datetime.now()))

    for epoch in range(epochs_num):

        print("{} Epoch Number: {}".format(datetime.now(), epoch+1))

        for batch in range(train_batches_epoch):

            img_batch, label_batch = get_batch(x_train, y_train, batch_size, batch)

            _, loss_value, acc_value = sess.run([optimizer, loss, accuracy],
                                                feed_dict={x: img_batch, y: label_batch, keep_prob: 0.5})

            if batch % display_step == 0:
                print("{}, Loss & Accuracy: {:.6f} {:.3f}".format(datetime.now(), loss_value, acc_value))

        if (epoch+1) % lr_decay_step == 0:
            learn_rate *= lr_decay_rate

        print("{} Start validation".format(datetime.now()))
        test_accuracy = 0.
        test_count = 0
        for _ in range(val_batches_epoch):
            img_batch, label_batch = get_batch(x_val, y_val, batch_size, _)
            acc = sess.run(accuracy, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.})

            test_accuracy += acc
            test_count += 1

        test_accuracy /= test_count

        print("{} Validation Accuracy = {:.3f}".format(datetime.now(), test_accuracy))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'ckpt')
        # save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
