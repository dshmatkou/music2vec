import tensorflow as tf
import logging
from common.dataset_records import FeaturedRecord  # XXX: heavy link

logger = logging.getLogger(__name__)


FILTERS = 24


def build_simple_cnn(input, kernel_size):
    return tf.layers.conv2d(
        inputs=input,
        filters=FILTERS,
        kernel_size=kernel_size,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123)
    )


# add activations
def build_kernel_model(input):
    input = tf.cast(input, tf.float32)
    with tf.variable_scope('kernel'):
        cnn1 = build_simple_cnn(input, [3, 3])
        cnn2 = build_simple_cnn(input, [5, 5])
        cnn3 = build_simple_cnn(input, [1, 1])

        pool1 = tf.layers.max_pooling2d(
            cnn1, pool_size=[3, 3], strides=(1, 1), padding='same'
        )
        pool2 = tf.layers.max_pooling2d(
            cnn2, pool_size=[3, 3], strides=(1, 1), padding='same'
        )
        pool3 = tf.layers.max_pooling2d(
            cnn3, pool_size=[3, 3], strides=(1, 1), padding='same'
        )

        flat1 = tf.contrib.layers.flatten(pool1)
        flat2 = tf.contrib.layers.flatten(pool2)
        flat3 = tf.contrib.layers.flatten(pool3)

        all_features = tf.concat([flat1, flat2, flat3], axis=1)

        result = tf.layers.dense(
            all_features,
            200,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123)
        )

    return result


def build_simple_multilabel_loss(kernel_model, label, label_name):
    summaries = []

    if label is None:
        units = getattr(FeaturedRecord, label_name).shape[0]
    else:
        units = label.shape[1]

    with tf.variable_scope(label_name):
        pred = tf.layers.dense(
            inputs=kernel_model,
            units=units,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
            activation=tf.nn.sigmoid,
        )
        summaries.append(tf.summary.tensor_summary('prediction', pred))

        if label is None:
            return pred, None, None, summaries

        loss = tf.losses.sigmoid_cross_entropy(
            label,
            pred,
        )
        summaries.append(tf.summary.scalar('loss', loss))

        acc = tf.metrics.accuracy(
            labels=label,
            predictions=pred,
        )
        summaries.append(tf.summary.tensor_summary('accuracy', acc))
    return pred, loss, acc, summaries


def build_simple_logit_loss(kernel_model, label, label_name):
    summaries = []

    if label is None:
        units = getattr(FeaturedRecord, label_name).shape[0]
    else:
        units = label.shape[1]

    with tf.variable_scope(label_name):
        pred = tf.layers.dense(
            inputs=kernel_model,
            units=units,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
            activation=tf.nn.sigmoid,
        )
        summaries.append(tf.summary.tensor_summary('prediction', pred))

        if label is None:
            return pred, None, None, summaries

        loss = tf.reduce_sum(
            tf.abs(
                tf.clip_by_value(label - pred, 1e-6, 1.0)
            )
        )
        summaries.append(tf.summary.scalar('loss', loss))

        acc = tf.metrics.accuracy(
            labels=label,
            predictions=pred,
        )
        summaries.append(tf.summary.tensor_summary('accuracy', acc))
    return pred, loss, acc, summaries


def build_simple_cat_loss(kernel_model, label, label_name):
    summaries = []

    if label is None:
        units = getattr(FeaturedRecord, label_name).shape[0]
    else:
        units = label.shape[1]

    with tf.variable_scope(label_name):
        pred = tf.layers.dense(
            inputs=kernel_model,
            units=units,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
            activation=tf.nn.sigmoid,
        )
        summaries.append(tf.summary.tensor_summary('prediction', pred))

        if label is None:
            return pred, None, None, summaries

        loss = tf.reduce_sum(
            tf.abs(
                tf.clip_by_value(label - pred, 1e-6, 1.0)
            )
        )
        summaries.append(tf.summary.scalar('loss', loss))

        acc = tf.metrics.accuracy(
            labels=label,
            predictions=pred,
        )
        summaries.append(tf.summary.tensor_summary('accuracy', acc))
    return pred, loss, acc, summaries


METRICS = {
    # label, metric
    'genres_all': build_simple_multilabel_loss,
    'genres_top': build_simple_multilabel_loss,
    # 'release_decade': build_simple_cat_loss,
    # 'acousticness': build_simple_logit_loss,
    # 'danceability': build_simple_logit_loss,
    # 'energy': build_simple_logit_loss,
    # 'instrumentalness': build_simple_logit_loss,
    # 'speechiness': build_simple_logit_loss,
    # 'happiness': build_simple_logit_loss,
    # 'artist_location': build_simple_cat_loss,
}


def model_fn(features, labels, mode):
    with tf.variable_scope('model'):

        model = build_kernel_model(features['feature'])

        losses = []
        accs = {}
        summaries = []
        predictions = {}

        for label_name, metric in METRICS.items():
            if mode == tf.estimator.ModeKeys.TRAIN and label_name not in labels:
                logger.warning('No label %s in labels', label_name)
                continue

            pred, loss, acc, lsum = metric(
                model,
                (labels or {}).get(label_name),
                label_name
            )
            predictions[label_name] = pred
            losses.append(loss)
            accs[label_name] = acc
            summaries.extend(lsum)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions['vector'] = model
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions
            )

        with tf.variable_scope('losses/total'):
            total_loss = tf.math.reduce_sum(losses)
            summaries.append(
                tf.summary.scalar('total_loss', total_loss)
            )
            # losses.append(total_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=0.1,
                    epsilon=0.1,
                )
                optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 1)
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
                loss=total_loss,
                train_op=tf.group(*training_ops),
                training_hooks=[summary_hook],
            )

        if tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=accs
            )
