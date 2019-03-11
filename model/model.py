import tensorflow as tf


FILTERS = 64


def build_simple_cnn(input, kernel_size):
    return tf.layers.conv2d(
        inputs=input,
        filters=FILTERS,
        kernel_size=kernel_size,
        padding='same',
        activation=tf.nn.relu,
    )


def build_kernel_model(features):
    x = features['feature']
    with tf.variable_scope("kernel_model"):
        cnn1 = build_simple_cnn(x, [5, 5])
        cnn2 = build_simple_cnn(x, [3, 3])
        cnn3 = build_simple_cnn(x, [1, 1])

        pool1 = tf.layers.max_pooling2d(cnn1, pool_size=[2, 2], strides=2)
        pool2 = tf.layers.max_pooling2d(cnn2, pool_size=[2, 2], strides=2)
        pool3 = tf.layers.max_pooling2d(cnn3, pool_size=[2, 2], strides=2)

        flat = tf.contrib.layers.flatten([pool1, pool2, pool3])
    return flat


def model_fn(features, labels, mode):
    model = build_kernel_model(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=model
        )

    # fixme
    # build simple classifiers and linear regressions over model
