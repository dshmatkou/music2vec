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
    pred = tf.layers.dense(
        inputs=kernel_model,
        units=label.shape[1],
        activation=tf.nn.relu,
    )
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
    summaries = [
        tf.summary.scalar('multilabel_loss/{}'.format(label_name), loss),
        tf.summary.scalar('multilabel_accuracy/{}'.format(label_name), acc[0]),
    ]
    return loss, acc[0], summaries


def build_simple_logit_loss(kernel_model, labels, label_name):
    label = labels[label_name]
    pred = tf.layers.dense(inputs=kernel_model, units=1, activation=None)
    loss = tf.losses.mean_squared_error(
        labels=label,
        predictions=pred,
    )
    acc = tf.metrics.accuracy(
        labels=label,
        predictions=pred,
    )
    summaries = [
        tf.summary.scalar('logit_loss/{}'.format(label_name), loss),
        tf.summary.scalar('logit_accuracy/{}'.format(label_name), acc[0]),
    ]
    return loss, acc[0], summaries


def build_simple_cat_loss(kernel_model, labels, label_name):
    label = labels[label_name]
    pred = tf.layers.dense(
        inputs=kernel_model,
        units=label.shape[1],
        activation=tf.nn.relu,
    )
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=label,
        logits=pred,
    )
    acc = tf.metrics.accuracy(
        labels=tf.argmax(label, axis=1),
        predictions=tf.argmax(pred, axis=1),
    )
    summaries = [
        tf.summary.scalar('category_loss/{}'.format(label_name), loss),
        tf.summary.scalar('category_accuracy/{}'.format(label_name), acc[0]),
    ]
    return loss, acc[0], summaries


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
        summaries = []
        if 'genres_all' in labels:
            loss, acc, label_summaries = build_simple_multilabel_loss(model, labels, 'genres_all')
            losses.append(loss)
            accs.append(acc)
            summaries.extend(label_summaries)
        if 'genres' in labels:
            loss, acc, label_summaries = build_simple_multilabel_loss(model, labels, 'genres')
            losses.append(loss)
            accs.append(acc)
            summaries.extend(label_summaries)
        # if 'date_released' in labels:
        #     loss, acc, label_summaries = build_simple_logit_loss(model, labels, 'date_released')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)
        # if 'acousticness' in labels:
        #     loss, acc, label_summaries = build_simple_logit_loss(model, labels, 'acousticness')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)
        # if 'danceability' in labels:
        #     loss, acc, label_summaries = build_simple_logit_loss(model, labels, 'danceability')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)
        # if 'energy' in labels:
        #     loss, acc, label_summaries = build_simple_logit_loss(model, labels, 'energy')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)
        # if 'instrumentalness' in labels:
        #     loss, acc, label_summaries = build_simple_logit_loss(model, labels, 'instrumentalness')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)
        # if 'speechiness' in labels:
        #     loss, acc, label_summaries = build_simple_logit_loss(model, labels, 'speechiness')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)
        # if 'valence' in labels:
        #     loss, acc, label_summaries = build_simple_logit_loss(model, labels, 'valence')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)
        # if 'artist_location' in labels:
        #     loss, acc, label_summaries = build_simple_cat_loss(model, labels, 'artist_location')
        #     losses.append(loss)
        #     accs.append(acc)
        #     summaries.extend(label_summaries)

    with tf.variable_scope('optimizer'):
        general_loss = tf.math.reduce_sum(losses)
        summaries.append(
            tf.summary.scalar('total_loss', general_loss)
        )
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=0.01,
                epsilon=0.01,
            )
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer,
                5,
            )
            training_ops = [
                optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step(),
                )
                for loss in losses
            ]
            summary_hook = tf.train.SummarySaverHook(
                1,
                output_dir='/tmp/music2vec_summary',
                summary_op=tf.summary.merge(summaries)
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=general_loss,
                train_op=tf.group(*training_ops),
                training_hooks=[summary_hook],
            )

    with tf.variable_scope('accuracy'):
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=general_loss,
            eval_metric_ops={
                'accuracy': sum(accs),
            }
        )

