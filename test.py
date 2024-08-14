import os, inspect, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import tensorflow as tf

import source.datamanager as dman
import source.neuralnet as nn
import source.tf_process as tfp

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CKPT_DIR = PACK_PATH+'/Checkpoint'

def test():

    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except: pass

    dataset = dman.Dataset(normalize=FLAGS.datnorm)
    neuralnet = nn.MemAE(height=dataset.height, width=dataset.width, channel=dataset.channel, leaning_rate=FLAGS.lr, ckpt_dir=CKPT_DIR)

    # Checkpoint 복원 설정
    vars_to_restore = {name: var for name, var in zip(neuralnet.name_bank, neuralnet.params_trainable)}
    vars_to_restore["optimizer"] = neuralnet.optimizer

    checkpoint = tf.train.Checkpoint(**vars_to_restore)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, CKPT_DIR, max_to_keep=1)

    # 체크포인트 복원
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print(f"Checkpoint restored from {checkpoint_manager.latest_checkpoint}")
    else:
        print("No checkpoint found. Exiting test.")
        return

    # 모델 테스트 수행
    tfp.test(neuralnet=neuralnet, dataset=dataset, batch_size=FLAGS.batch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--lr', type=int, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=1000, help='Training epoch')
    parser.add_argument('--batch', type=int, default=32, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    test()
