import tensorflow as tf
import wave
import numpy as np
import pickle
import json
import random
from pathlib import Path
from tqdm import tqdm
import datetime


def wavs2data(fnames):
    xs = []
    for fname in fnames:
        print(fname)
        wav = wave.open(str(fname))
        x = wav.readframes(wav.getnframes())
        x = np.frombuffer(x, dtype=np.uint8)
        sr = wav.getframerate()
        sampwidth = 2**(wav.getsampwidth() * 8 - 1)
        xs.append(x)
    x = np.hstack(xs)
    xmin = x.min()
    x = x - xmin
    vocab = sorted(set(x))
    return x, sr, sampwidth, xmin, vocab


def bytes2audio(x, xmin, s):
    return ((np.array(x) + xmin).astype(np.float32) - s) / s


def build_model(vocab_size, embedding_dim, rnn_units, batch_size, rnn_layers=1, batchnorm=False):
    layers = [tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None])]
    for layer in range(rnn_layers):
        layers.append(tf.keras.layers.GRU(rnn_units,
                                          return_sequences=True,
                                          stateful=True,
                                          recurrent_initializer='glorot_uniform'))
        if batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())

    layers.append(tf.keras.layers.Dense(vocab_size))
    model = tf.keras.Sequential(layers)
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * (1 + tf.math.sin(epoch / 40 * 6 * 6.28) * 0.5) * tf.math.exp(-0.1)


def generate_text(model, input_eval, num_generate=10000, temperature=1.0):
    input_eval = tf.expand_dims(input_eval, 1)
    generated = [input_eval.numpy().T]
    model.reset_states()
    for i in tqdm(range(num_generate)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 1)
        predictions = predictions / temperature
        predicted = tf.random.categorical(predictions, num_samples=1)[:, 0].numpy()
        input_eval = tf.expand_dims(predicted, 1)
        generated.append(predicted)
    return generated


def audio2data(x, seq_length, batch_size):
    # Create training examples / targets
    audio_samples_dataset = tf.data.Dataset.from_tensor_slices(x)
    sequences = audio_samples_dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    return dataset


def train(run_name, wavs_path, embedding_dim, rnn_units, layers, batchnorm, epochs, lr, seq_length, batch_size=512, seed_value=12345, log_dir='tb_logs'):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    x, sr, sampwidth, xmin, vocab = wavs2data(list(Path(wavs_path).rglob('*.wav')))
    dataset = audio2data(x, seq_length, batch_size)

    vocab_size = max(vocab) + 1

    config = {'vocab_size': int(vocab_size), 'embedding_dim': embedding_dim,
              'rnn_units': rnn_units, 'layers': layers, 'sr': sr, 'sampwidth': sampwidth,
              'xmin': int(xmin), 'batchnorm': batchnorm, 'batch_size': batch_size,
              'seq_length': seq_length, 'lr': lr, 'seed_value': seed_value, 'epochs': epochs}

    run_name = Path(run_name)
    run_name.mkdir(exist_ok=True, parents=True)

    # display(Audio(bytes2audio(x,xmin,sampwidth),rate=sr) )

    np.save(Path(run_name, 'data.npy'), x)

    with open(Path(run_name, 'model_function.pkl'), 'wb') as fp:
        pickle.dump(build_model, fp)

    with open(Path(run_name, 'parameters.json'), 'w') as fp:
        json.dump(config, fp)

    model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units,
                        batch_size=batch_size, rnn_layers=layers, batchnorm=batchnorm)

    run_specs = '-'.join([f'{k}_{v}' for k, v in config.items()])

    print(min(vocab), max(vocab), len(x))
    print(run_specs)

    log_dir = Path('tb_logs', run_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_ ' + run_specs)

    fit(model, dataset, epochs, lr, run_name, log_dir)

def fit(model, dataset, epochs, lr, checkpoint_dir, log_dir, initial_epoch=0):

    checkpoint_prefix = Path(checkpoint_dir, 'ckpt', 'epochv_bn_{epoch:03d}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_prefix),
                                                             save_weights_only=True, save_freq='epoch', period=6)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, profile_batch=0)

    # lr_sched_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks = [tensorboard_callback, checkpoint_callback]

    for input_example_batch, target_example_batch in dataset.take(1):
        model(input_example_batch)

    opt = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss=loss)

    print(model.summary())

    history = model.fit(dataset, epochs=epochs, shuffle=True, callbacks=callbacks, verbose=2, initial_epoch=initial_epoch)

    with open(Path(checkpoint_dir, 'history.pkl'), 'wb') as fp:
        pickle.dump(history.history, fp)


# def train(checkpoint_dir, wavs_path, embedding_dim, rnn_units, layers, batchnorm, epochs, lr, seq_length, batch_size=512, seed_value=12345, log_dir='tb_logs'):
#     tf.keras.backend.clear_session()
#     tf.random.set_seed(seed_value)
#     random.seed(seed_value)
#     np.random.seed(seed_value)

#     x, sr, sampwidth, xmin, vocab = wavs2data(list(Path(wavs_path).rglob('*.wav')))
#     vocab_size = max(vocab) + 1

#     config = {'vocab_size': int(vocab_size), 'embedding_dim': embedding_dim,
#               'rnn_units': rnn_units, 'layers': layers, 'sr': sr, 'sampwidth': sampwidth,
#               'xmin': int(xmin), 'batchnorm': batchnorm, 'batch_size': batch_size,
#               'seq_length': seq_length, 'lr': lr, 'seed_value': seed_value}

#     print(min(vocab), max(vocab), len(x))
#     print(config)

#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoint_dir.mkdir(exist_ok=True, parents=True)

#     # display(Audio(bytes2audio(x,xmin,sampwidth),rate=sr) )

#     np.save(Path(checkpoint_dir, 'data.npy'), x)

#     with open(Path(checkpoint_dir, 'model_function.pkl'), 'wb') as fp:
#         pickle.dump(build_model, fp)

#     with open(Path(checkpoint_dir, 'parameters.json'), 'w') as fp:
#         json.dump(config, fp)

#     audio_samples_dataset = tf.data.Dataset.from_tensor_slices(x)
#     sequences = audio_samples_dataset.batch(seq_length + 1, drop_remainder=True)
#     dataset = sequences.map(split_input_target)
#     BUFFER_SIZE = 10000
#     dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

#     checkpoint_prefix = str(Path(checkpoint_dir, 'ckpt', 'epochv_bn_{epoch:03d}'))
#     checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True, save_freq=6)

#     run_specs = '-'.join(['{k}_{v}'.format(k=k, v=v) for k, v in config.items()])

#     logdir = Path('tb_logs', run_specs)
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, profile_batch=0)

#     # lr_sched_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#     callbacks = [tensorboard_callback, checkpoint_callback]

#     model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units,
#                         batch_size=batch_size, rnn_layers=layers, batchnorm=batchnorm)

#     for input_example_batch, target_example_batch in dataset.take(1):
#         model(input_example_batch)

#     opt = tf.keras.optimizers.Adam(lr)
#     model.compile(optimizer=opt, loss=loss)

#     print(model.summary())

#     history = model.fit(dataset, epochs=epochs, shuffle=True, callbacks=callbacks, verbose=2)

#     with open(Path(checkpoint_dir, 'history.pkl'), 'wb') as fp:
#         pickle.dump(history.history, fp)
