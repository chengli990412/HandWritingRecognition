import numpy as np
import os
import gzip
import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, QThread,QMutex
import time
from tensorflow import keras
from keras import layers, optimizers, datasets


class DeepLearning(QThread):
    # 自定义信号，负责主界面Log栏刷新
    textWritten = pyqtSignal(str)
    # 自定义信号，负责主界面Mat绘画
    drawMat = pyqtSignal(float,float,float,float)

    def __init__(self,parent=None):
        super(DeepLearning,self).__init__(parent)
        self.mutex=QMutex()


    # 数据集加载
    def run(self):
        self.mutex.lock()
        try:
            # 训练时间统计
            t = time.perf_counter()

            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.device("/gpu:0")

            files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz']
            paths = []
            for fname in files:
                paths.append(os.path.join('MNIST/', fname))
            if len(paths) == 0:
                return '路径异常，请检测路径设置是否正确'

            with gzip.open(paths[0], 'rb') as lbpath:
                train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open(paths[1], 'rb') as imgpath:
                train_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)

            with gzip.open(paths[2], 'rb') as lbpath:
                test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open(paths[3], 'rb') as imgpath:
                test_image = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_label), 28, 28)

            Model = keras.Sequential([layers.Dense(256, activation='relu'),
                                      layers.Dense(128, activation='relu'),
                                      layers.Dense(10)])
            Model.build(input_shape=(None, 28 * 28))
            Model.summary()
            x = tf.convert_to_tensor(train_images, dtype=tf.float32) / 255
            db = tf.data.Dataset.from_tensor_slices((x, train_labels))
            db = db.batch(100).repeat(20)
            optimizer = optimizers.SGD(lr=0.01)
            acc_meter = keras.metrics.Accuracy()
            summary_writer = tf.summary.create_file_writer('tf_log')
            for step, (xx, yy) in enumerate(db):
                with tf.GradientTape() as tape:
                    # 图像样本大小重置(-1, 28*28)
                    xx = tf.reshape(xx, (-1, 28 * 28))
                    # 获取输出
                    out = Model(xx)
                    # 实际标签转为onehot编码
                    y_onehot = tf.one_hot(yy, depth=10)
                    # 计算误差
                    loss = tf.square(out - y_onehot)
                    loss = tf.reduce_sum(loss / xx.shape[0])
                    # 更新准备率
                    acc_meter.update_state(tf.argmax(out, axis=1), yy)
                    # 更新梯度参数
                    grads = tape.gradient(loss, Model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, Model.trainable_variables))
                    # 绘图
                    if step % 100 == 0:
                        self.drawMat.emit(step, float(loss),step,float(acc_meter.result().numpy()) )

                    # 参数存储，便于查看曲线图
                    with summary_writer.as_default():
                        tf.summary.scalar('train-loss', float(loss), step=step)
                        tf.summary.scalar('test-acc', acc_meter.result().numpy(), step=step)

                    if step % 1000 == 0:
                        mes = "【训练第:"+str(step)+"次】\n"+"损失函数："+str(round(float(loss),3))+"\n"+"模型识别准确率："+str(acc_meter.result().numpy())
                        self.textWritten.emit(mes)
                        acc_meter.reset_states()

            self.textWritten.emit(f' 【训练完成】，总共耗时:{time.perf_counter() - t:.2f}s')
        except Exception as ex:
            print(ex)
        self.mutex.unlock()

def SaveModel(model):
    model.save('Model/')
