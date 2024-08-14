import os, inspect, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
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
    except: pass

    male_dir = './spectrogram/Male'
    female_dir = './spectrogram/Female'
    dataset = dman.Dataset(normalize=FLAGS.datnorm)
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

    # 모델 학습
    tfp.training(neuralnet=neuralnet, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, normalize=True)

    # 학습 후 체크포인트 저장
    checkpoint_manager.save()
    print(f"Checkpoint saved at {CKPT_DIR}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=1000, help='Training epoch')
    parser.add_argument('--batch', type=int, default=32, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    train()
