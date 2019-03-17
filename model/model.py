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


def build_kernel_model(feature):
    with tf.variable_scope('kernel_model'):
        cnn1 = build_simple_cnn(feature, [5, 5])
        cnn2 = build_simple_cnn(feature, [3, 3])
        cnn3 = build_simple_cnn(feature, [1, 1])

        pool1 = tf.layers.max_pooling2d(cnn1, pool_size=[2, 2], strides=2)
        pool2 = tf.layers.max_pooling2d(cnn2, pool_size=[2, 2], strides=2)
        pool3 = tf.layers.max_pooling2d(cnn3, pool_size=[2, 2], strides=2)

        flat1 = tf.contrib.layers.flatten(pool1)
        flat2 = tf.contrib.layers.flatten(pool2)
        flat3 = tf.contrib.layers.flatten(pool3)

        result = tf.concat([flat1, flat2, flat3], axis=1)
    return result


def build_simple_multilabel_loss(kernel_model, labels, label_name):
    label = labels[label_name]
    pred = tf.layers.dense(inputs=kernel_model, units=label.shape[1])
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label,
        logits=pred,
    )
    # to cast to scalar
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    acc = tf.metrics.accuracy(
        labels=tf.round(tf.nn.sigmoid(label)),
        predictions=pred,
    )
    return loss, acc


def build_simple_logit_loss(kernel_model, labels, label_name):
    label = labels[label_name]
    pred = tf.layers.dense(inputs=kernel_model, units=1)
    loss = tf.losses.mean_squared_error(
        labels=label,
        predictions=pred,
    )
    acc = tf.metrics.accuracy(
        labels=label,
        predictions=pred,
    )
    return loss, acc


def build_simple_cat_loss(kernel_model, labels, label_name):
    label = labels[label_name]
    pred = tf.layers.dense(inputs=kernel_model, units=label.shape[1])
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=label,
        logits=pred,
    )
    acc = tf.metrics.accuracy(
        labels=tf.argmax(label, axis=1),
        predictions=tf.argmax(pred, axis=1),
    )
    return loss, acc


def model_fn(features, labels, mode):
    model = build_kernel_model(features['feature'])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=model
        )

    with tf.variable_scope('losses'):
        losses = []
        accs = []
        if 'genres_all' in labels:
            loss, acc = build_simple_multilabel_loss(model, labels, 'genres_all')
            losses.append(loss)
            accs.append(acc)
        if 'genres' in labels:
            loss, acc = build_simple_multilabel_loss(model, labels, 'genres')
            losses.append(loss)
            accs.append(acc)
        if 'date_released' in labels:
            loss, acc = build_simple_logit_loss(model, labels, 'date_released')
            losses.append(loss)
            accs.append(acc)
        if 'acousticness' in labels:
            loss, acc = build_simple_logit_loss(model, labels, 'acousticness')
            losses.append(loss)
            accs.append(acc)
        if 'danceability' in labels:
            loss, acc = build_simple_logit_loss(model, labels, 'danceability')
            losses.append(loss)
            accs.append(acc)
        if 'energy' in labels:
            loss, acc = build_simple_logit_loss(model, labels, 'energy')
            losses.append(loss)
            accs.append(acc)
        if 'instrumentalness' in labels:
            loss, acc = build_simple_logit_loss(model, labels, 'instrumentalness')
            losses.append(loss)
            accs.append(acc)
        if 'speechiness' in labels:
            loss, acc = build_simple_logit_loss(model, labels, 'speechiness')
            losses.append(loss)
            accs.append(acc)
        if 'valence' in labels:
            loss, acc = build_simple_logit_loss(model, labels, 'valence')
            losses.append(loss)
            accs.append(acc)
        if 'artist_location' in labels:
            loss, acc = build_simple_cat_loss(model, labels, 'artist_location')
            losses.append(loss)
            accs.append(acc)

    with tf.variable_scope('optimizer'):
        general_loss = sum(losses)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(
                loss=general_loss,
                global_step=tf.train.get_global_step(),
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=general_loss,
                train_op=train_op,
            )

    with tf.variable_scope('accuracy'):
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=general_loss,
            eval_metric_ops={
                'accuracy': sum(accs),
            }
        )

