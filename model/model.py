import tensorflow as tf
import logging
from common.dataset_records import FeaturedRecord  # XXX: heavy link

logger = logging.getLogger(__name__)


FILTERS = 32


def build_simple_cnn(input, kernel_size):
    return tf.layers.conv2d(
        inputs=input,
        filters=FILTERS,
        kernel_size=kernel_size,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
    )


# add activations
def build_kernel_model(input, mode):
    input = tf.cast(input, tf.float32)
    with tf.variable_scope('kernel'):
        # tower [3, 3]
        cnn1 = build_simple_cnn(input, [1, 1])
        cnn1 = tf.layers.max_pooling2d(
            cnn1, pool_size=[3, 3], strides=1
        )
        cnn1 = tf.layers.dropout(
            cnn1, 0.3,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )
        cnn1 = build_simple_cnn(cnn1, [3, 3])
        cnn1 = tf.layers.max_pooling2d(
            cnn1, pool_size=[3, 3], strides=1,
        )
        cnn1 = tf.layers.dropout(
            cnn1, 0.3,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )

        # tower [5, 5]
        cnn2 = build_simple_cnn(input, [1, 1])
        cnn2 = tf.layers.max_pooling2d(
            cnn2, pool_size=[3, 3], strides=1
        )
        cnn2 = tf.layers.dropout(
            cnn2, 0.3,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )
        cnn2 = build_simple_cnn(cnn2, [5, 5])
        cnn2 = tf.layers.max_pooling2d(
            cnn2, pool_size=[3, 3], strides=1,
        )
        cnn2 = tf.layers.dropout(
            cnn2, 0.3,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )

        # tower [1, 1]
        cnn3 = tf.layers.max_pooling2d(
            input, pool_size=[3, 3], strides=1
        )
        cnn3 = tf.layers.dropout(
            cnn3, 0.3,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )
        cnn3 = build_simple_cnn(cnn3, [1, 1])
        cnn3 = tf.layers.max_pooling2d(
            cnn3, pool_size=[3, 3], strides=1,
        )
        cnn3 = tf.layers.dropout(
            cnn3, 0.3,
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )

        flat1 = tf.contrib.layers.flatten(cnn1)
        flat2 = tf.contrib.layers.flatten(cnn2)
        flat3 = tf.contrib.layers.flatten(cnn3)

        all_features = tf.concat([flat1, flat2, flat3], axis=1)
        result = tf.layers.dense(
            all_features, 200,
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
        loss_value = tf.layers.dense(
            inputs=kernel_model,
            units=units,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
        )
        pred = tf.nn.sigmoid(loss_value)
        summaries.append(tf.summary.tensor_summary('prediction', pred))

        if label is None:
            return pred, None, None, summaries

        binary_predictions = tf.to_float(tf.greater(pred, 0.5))
        weights = tf.abs(binary_predictions - label) + tf.constant(0.1)

        loss = tf.losses.sigmoid_cross_entropy(
            tf.clip_by_value(label, 1e-3, 0.999),
            tf.clip_by_value(loss_value, 1e-3, 0.999),
            reduction=tf.losses.Reduction.SUM,
            weights=weights,
        )
        summaries.append(tf.summary.scalar('loss', loss))

        acc = tf.metrics.precision(
            labels=label,
            predictions=binary_predictions,
        )
        scalar_acc = tf.reduce_mean(acc[0])
        summaries.append(tf.summary.scalar('accuracy', scalar_acc))
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
            units=250,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
            activation=tf.nn.relu,
        )
        pred = tf.layers.dense(
            inputs=pred,
            units=units,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
        )
        summaries.append(tf.summary.tensor_summary('prediction', pred))

        if label is None:
            return pred, None, None, summaries

        loss = tf.losses.huber_loss(
            label,
            pred,
            reduction=tf.losses.Reduction.SUM
        )
        summaries.append(tf.summary.scalar('loss', loss))

        acc = tf.metrics.mean_absolute_error(
            labels=label,
            predictions=pred,
        )
        mse = tf.metrics.mean_squared_error(
            labels=label,
            predictions=pred,
        )

        scalar_acc = tf.reduce_mean(acc[0])
        scalar_mse = tf.reduce_mean(mse[0])
        summaries.append(tf.summary.scalar('mean_absolute_error', scalar_acc))
        summaries.append(tf.summary.scalar('mean_squared_error', scalar_mse))
    return pred, loss, mse, summaries


def build_simple_cat_loss(kernel_model, label, label_name):
    summaries = []

    if label is None:
        units = getattr(FeaturedRecord, label_name).shape[0]
    else:
        units = label.shape[1]

    with tf.variable_scope(label_name):
        loss_value = tf.layers.dense(
            inputs=kernel_model,
            units=units,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=123),
            activation=tf.nn.sigmoid,
        )
        pred = tf.nn.softmax(loss_value)
        summaries.append(tf.summary.tensor_summary('prediction', pred))

        if label is None:
            return pred, None, None, summaries

        loss = tf.losses.softmax_cross_entropy(
            tf.clip_by_value(label, 1e-3, 0.999),
            tf.clip_by_value(loss_value, 1e-3, 0.999),
            weights=0.15,
            reduction=tf.losses.Reduction.SUM,
        )
        summaries.append(tf.summary.scalar('loss', loss))

        acc = tf.metrics.accuracy(
            labels=tf.argmax(label),
            predictions=tf.argmax(pred),
        )
        scalar_acc = tf.reduce_mean(acc[0])
        summaries.append(tf.summary.scalar('accuracy', scalar_acc))
    return pred, loss, acc, summaries


METRICS = {
    # label, metric
    # 'genres_all': build_simple_multilabel_loss,
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

        model = build_kernel_model(features['feature'], mode)

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

        with tf.variable_scope('total'):
            total_loss = tf.math.reduce_sum(losses)
            summaries.append(
                tf.summary.scalar('loss', total_loss)
            )
            total_accuracy = tf.reduce_mean(list(accs.values()))
            summaries.append(
                tf.summary.scalar('accuracy', total_accuracy)
            )
            # losses.append(total_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=0.005,
                )
                optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 2)
                training_ops = [
                    optimizer.minimize(
                        loss=loss,
                        global_step=tf.train.get_global_step(),
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
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
