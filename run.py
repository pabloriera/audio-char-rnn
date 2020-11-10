import tensorflow as tf
from audio_char_rnn import train, bytes2audio, generate_text, fit, audio2data
from pathlib import Path
import pickle
import json
import time
import numpy as np
import argparse
import soundfile as sf
import datetime
from IPython import embed

def main(run_name, train_, continue_, generate, length=10, epochs=100, batch_size=64, generate_from_ckpt=None, generate_seed='data'):

    wavs_path = 'wavs/8bit'
    run_name = Path(run_name)

    params = {'layers': 2, 'rnn_units': 1024, 'batchnorm': False, 'seed_value': 12345,
              'lr': 0.0008, 'embedding_dim': 256, 'seq_length': 200, 'batch_size': 256}

    if train_:
        train(run_name, wavs_path, epochs=epochs, **params)

    if continue_:
        with open(Path(run_name, 'model_function.pkl'), 'rb') as fp:
            build_model = pickle.load(fp)

        with open(Path(run_name, 'parameters.json'), 'r') as fp:
            config = json.load(fp)

        vocab_size = config['vocab_size']
        embedding_dim = config['embedding_dim']
        sampwidth = config['sampwidth']
        rnn_units = config['rnn_units']
        xmin = config['xmin']
        sr = config['sr']
        lr = config['lr']
        seq_length = config['seq_length']
        batchnorm = config['batchnorm']
        batch_size = config['batch_size']
        prev_epochs = config['epochs']
        config['epochs'] = epochs + prev_epochs

        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=batch_size, batchnorm=batchnorm)
        model.load_weights(tf.train.latest_checkpoint(str(run_name / 'ckpt')))
        model.build(tf.TensorShape([batch_size, None]))

        x = np.load(run_name / 'data.npy')
        dataset = audio2data(x, seq_length, batch_size)

        run_specs = '-'.join([f'{k}_{v}' for k, v in config.items()])
        log_dir = Path('tb_logs', run_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_ ' + run_specs)

        fit(model, dataset, epochs, lr, run_name, log_dir, initial_epoch=prev_epochs)

        with open(Path(run_name, 'parameters.json'), 'w') as fp:
            json.dump(config, fp)

    if generate:
        with open(Path(run_name, 'model_function.pkl'), 'rb') as fp:
            build_model = pickle.load(fp)

        with open(Path(run_name, 'parameters.json'), 'r') as fp:
            config = json.load(fp)

        print(config)
        vocab_size = config['vocab_size']
        embedding_dim = config['embedding_dim']
        sampwidth = config['sampwidth']
        rnn_units = config['rnn_units']
        xmin = config['xmin']
        sr = config['sr']
        # lr = config['lr']
        seq_length = config['seq_length']
        # batchnorm = config['batchnorm']
        # batch_size = config['batch_size']

        samps = int(length * sr)
        print('Generating', samps, 'samples')

        if generate_seed == 'linspace':
            generate_seed = np.uint8(np.linspace(0, vocab_size, batch_size))
        # elif generate_seed == 'data':
        #     x = np.load(run_name / 'data.npy')
        #     ix = np.random.randint(0, len(x) - 1 - seq_length)
        #     generate_seed = x[ix:ix + seq_length]
        #     generate_seed = generate_seed[np.newaxis, :]
        #     batch_size = 1
        elif generate_seed == 'data':
            generate_seed = []
            x = np.load(run_name / 'data.npy')
            for i in range(batch_size):
                ix = np.random.randint(0, len(x) - 1)
                generate_seed.append(x[ix])
            generate_seed = np.array(generate_seed)

        if generate_from_ckpt is None:
            ckpt = Path(tf.train.latest_checkpoint(run_name / 'ckpt'))
        else:
            ckpt = Path(generate_from_ckpt)

        print('Using chekpoint', ckpt)

        out = Path(run_name, 'wavs_epochs')
        Path(*out.parts).mkdir(exist_ok=True, parents=True)
        print(out)
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=batch_size, batchnorm=False)
        model.load_weights(str(ckpt))
        model.build(tf.TensorShape([batch_size, None]))
        tic = time.time()
        
        generated = generate_text(model, generate_seed, num_generate=samps)
        print(time.time() - tic)
        generated = np.vstack(generated).T
        for i, gen in enumerate(generated):
            audio = bytes2audio(gen, xmin, sampwidth)
            out = Path(run_name, 'wavs_epochs', ckpt.stem + '_' + str(samps) + '_' + str(i) + '.wav')
            sf.write(out, audio, sr)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('run_name', help='NAME', type=str)
    argparser.add_argument('--train', dest='train_', help='TRAIN', action='store_true', default=False)
    argparser.add_argument('--continue', dest='continue_', help='CONTINUE', action='store_true', default=False)
    argparser.add_argument('--generate', help='GENERATE', action='store_true', default=False)
    argparser.add_argument('--length', help='GENERATE', type=float, default=10)
    argparser.add_argument('--epochs', help='epochs', type=int, default=10)
    argparser.add_argument('--batch_size', help='batch_size', type=int, default=6)
    argparser.add_argument('--generate-from-ckpt', help='GENERATE', type=str, default=None)

    args = vars(argparser.parse_args())

    main(**args)
