import os, inspect, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
import source.datamanager as dman
import source.neuralnet as nn
import source.tf_process as tfp

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CKPT_DIR = PACK_PATH + '/Checkpoint'

def train():

    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except: 
        pass

    male_dir = './spectrogram/Male'
    female_dir = './spectrogram/Female'
    dataset = dman.Dataset(normalize=FLAGS.datnorm)
    print(dataset.channel)
    neuralnet = nn.MemAE(height=dataset.height, width=dataset.width, channel=dataset.channel, leaning_rate=FLAGS.lr, ckpt_dir=CKPT_DIR)

    # Checkpoint 설정
    vars_to_restore = {name: var for name, var in zip(neuralnet.name_bank, neuralnet.params_trainable)}
    vars_to_restore["optimizer"] = neuralnet.optimizer
    checkpoint = tf.train.Checkpoint(**vars_to_restore)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CKPT_DIR, max_to_keep=3)

    # 기존 체크포인트 복원
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Checkpoint restored from {checkpoint_manager.latest_checkpoint}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # Early Stopping 설정
    best_loss = float('inf')
    patience = FLAGS.patience
    min_delta = FLAGS.min_delta
    wait = 0

    # 모델 학습
    for epoch in range(FLAGS.epoch):
        avg_loss = tfp.training(neuralnet=neuralnet, dataset=dataset, epochs=1, batch_size=FLAGS.batch, normalize=True)

        # Early stopping 체크
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            wait = 0
            checkpoint_manager.save()  # 성능이 개선될 때마다 체크포인트 저장
            print(f"Checkpoint saved at {CKPT_DIR} for epoch {epoch + 1}")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Training complete.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=1000, help='Training epoch')
    parser.add_argument('--batch', type=int, default=64, help='Mini batch size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as an improvement')

    FLAGS, unparsed = parser.parse_known_args()

    train()
