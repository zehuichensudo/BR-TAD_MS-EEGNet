#%%
import os
import sys
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

import models
from preprocess import get_data

#%%
def draw_learning_curves(history, sub, results_path):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy - subject: ' + str(sub))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/subject_' + str(sub) + '_accuracy.png')
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss - subject: ' + str(sub))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/subject_' + str(sub) + '_loss.png')
    plt.close()

def draw_confusion_matrix(cf_matrix, sub, results_path, classes_labels):
    display_labels = classes_labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix of Subject: ' + sub )
    plt.savefig(results_path + '/subject_' + sub + '_confusion.png')
    plt.close()

def draw_performance_barChart(num_sub, metric, label, results_path):
    fig, ax = plt.subplots()
    x = list(range(1, num_sub+1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model '+ label + ' per subject')
    ax.set_ylim([0,1])
    plt.savefig(results_path + '/performance_' + label + '.png')
    plt.close()

#%% 训练过程
# 🚀 唯一修改点：为 train 函数增加了 use_14_channels 和 apply_bandpass 参数并透传
def train(dataset_conf, train_conf, results_path, use_br_tad=False, add_noise=False, snr_range=(-10, -2), use_14_channels=True, apply_bandpass=True):

    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    in_exp = time.time()
    best_models = open(results_path + "/best models.txt", "w")
    log_write = open(results_path + "/log.txt", "w")

    dataset = dataset_conf.get('name')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')

    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')
    from_logits = train_conf.get('from_logits')

    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    for sub in range(n_sub):

        print('\nTraining on subject ', sub+1)
        log_write.write( '\nTraining on subject '+ str(sub+1) +'\n')
        BestSubjAcc = 0
        bestTrainingHistory = []

        X_train, _, y_train_onehot, _, _, _ = get_data(
            data_path, sub, dataset, LOSO = LOSO, isStandard = isStandard,
            use_br_tad=use_br_tad, add_noise=add_noise, snr_range=snr_range,
            use_14_channels=use_14_channels, apply_bandpass=apply_bandpass)

        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(X_train, y_train_onehot, test_size=0.2, random_state=42)

        for run_idx in range(n_train):
            tf.random.set_seed(run_idx+1)
            np.random.seed(run_idx+1)

            in_run = time.time()

            filepath = results_path + '/saved models/run-{}'.format(run_idx+1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + '/subject-{}.h5'.format(sub+1)

            model = getModel(model_name, dataset_conf, from_logits)
            model.compile(loss=CategoricalCrossentropy(from_logits=from_logits), optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

            callbacks = [
                ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='min'),
                ReduceLROnPlateau(monitor="val_loss", factor=0.90, patience=20, verbose=0, min_lr=0.0001),
            ]
            history = model.fit(X_train, y_train_onehot, validation_data=(X_val, y_val_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            model.load_weights(filepath)
            y_pred = model.predict(X_val)

            if from_logits:
                y_pred = tf.nn.softmax(y_pred).numpy().argmax(axis=-1)
            else:
                y_pred = y_pred.argmax(axis=-1)

            labels = y_val_onehot.argmax(axis=-1)
            acc[sub, run_idx]  = accuracy_score(labels, y_pred)
            kappa[sub, run_idx] = cohen_kappa_score(labels, y_pred)

            out_run = time.time()
            info = 'Subject: {}   seed {}   time: {:.1f} m   '.format(sub+1, run_idx+1, ((out_run-in_run)/60))
            info = info + 'valid_acc: {:.4f}   valid_loss: {:.3f}'.format(acc[sub, run_idx], min(history.history['val_loss']))
            print(info)
            log_write.write(info +'\n')

            if(BestSubjAcc < acc[sub, run_idx]):
                 BestSubjAcc = acc[sub, run_idx]
                 bestTrainingHistory = history
            tf.keras.backend.clear_session()

        best_run = np.argmax(acc[sub,:])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run+1, sub+1)+'\n'
        best_models.write(filepath)

        if (LearnCurves == True):
            draw_learning_curves(bestTrainingHistory, sub + 1, results_path)

    out_exp = time.time()

    head1 = head2 = '         '
    for sub in range(n_sub):
        head1 = head1 + 'sub_{}   '.format(sub+1)
        head2 = head2 + '-----   '
    head1 = head1 + '  average'
    head2 = head2 + '  -------'
    info = '\n---------------------------------\nValidation performance (acc %):'
    info = info + '\n---------------------------------\n' + head1 +'\n'+ head2
    for run in range(n_train):
        info = info + '\nSeed {}:  '.format(run+1)
        for sub in range(n_sub):
            info = info + '{:.2f}   '.format(acc[sub, run]*100)
        info = info + '  {:.2f}   '.format(np.average(acc[:, run])*100)
    info = info + '\n---------------------------------\nAverage acc - all seeds: '
    info = info + '{:.2f} %\n\nTrain Time  - all seeds: {:.1f}'.format(np.average(acc)*100, (out_exp-in_exp)/(60))
    info = info + ' min\n---------------------------------\n'
    print(info)
    log_write.write(info+'\n')

    best_models.close()
    log_write.close()


#%% 评估过程
def test(model, dataset_conf, results_path, allRuns = True, use_br_tad=False, add_noise=False, snr_range=(-10, -2), use_14_channels=True, apply_bandpass=True):
    log_write = open(results_path + "/log.txt", "a")

    dataset = dataset_conf.get('name')
    n_classes = dataset_conf.get('n_classes')
    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    LOSO = dataset_conf.get('LOSO')
    classes_labels = dataset_conf.get('cl_labels')

    runs = os.listdir(results_path+"/saved models")
    acc = np.zeros((n_sub, len(runs)))
    kappa = np.zeros((n_sub, len(runs)))
    cf_matrix = np.zeros([n_sub, len(runs), n_classes, n_classes])

    inference_time = 0
    for sub in range(n_sub):
        # 🚀 唯一修改点：将这俩新参数传给 get_data
        _, _, _, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO = LOSO, isStandard = isStandard,
            use_br_tad=use_br_tad, add_noise=add_noise, snr_range=snr_range,
            use_14_channels=use_14_channels, apply_bandpass=apply_bandpass)

        for seed in range(len(runs)):
            model.load_weights('{}/saved models/{}/subject-{}.h5'.format(results_path, runs[seed], sub+1))

            inf_start = time.time()
            y_pred = model.predict(X_test).argmax(axis=-1)
            inference_time = (time.time() - inf_start)/X_test.shape[0]

            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, seed]  = accuracy_score(labels, y_pred)
            kappa[sub, seed] = cohen_kappa_score(labels, y_pred)
            cf_matrix[sub, seed, :, :] = confusion_matrix(labels, y_pred, normalize='true')

    head1 = head2 = '                  '
    for sub in range(n_sub):
        head1 = head1 + 'sub_{}   '.format(sub+1)
        head2 = head2 + '-----   '
    head1 = head1 + '  average'
    head2 = head2 + '  -------'
    info = '\n---------------------------------\nTest performance (acc & k-score):\n'
    info = info + '---------------------------------\n' + head1 +'\n'+ head2
    for run in range(len(runs)):
        info = info + '\nSeed {}: '.format(run+1)
        info_acc = '(acc %)   '
        info_k = '        (k-sco)   '
        for sub in range(n_sub):
            info_acc = info_acc + '{:.2f}   '.format(acc[sub, run]*100)
            info_k = info_k + '{:.3f}   '.format(kappa[sub, run])
        info_acc = info_acc + '  {:.2f}   '.format(np.average(acc[:, run])*100)
        info_k = info_k + '  {:.3f}   '.format(np.average(kappa[:, run]))
        info = info + info_acc + '\n' + info_k
    info = info + '\n----------------------------------\nAverage - all seeds (acc %): '
    info = info + '{:.2f}\n                    (k-sco): '.format(np.average(acc)*100)
    info = info + '{:.3f}\n\nInference time: {:.2f}'.format(np.average(kappa), inference_time * 1000)
    info = info + ' ms per trial\n----------------------------------\n'
    print(info)
    log_write.write(info+'\n')

    draw_performance_barChart(n_sub, acc.mean(1), 'Accuracy', results_path)
    draw_performance_barChart(n_sub, kappa.mean(1), 'k-score', results_path)
    draw_confusion_matrix(cf_matrix.mean((0,1)), 'All', results_path, classes_labels)
    log_write.close()


#%% 模型工厂
def getModel(model_name, dataset_conf, from_logits = False):

    n_classes = dataset_conf.get('n_classes')
    n_channels = dataset_conf.get('n_channels')
    in_samples = dataset_conf.get('in_samples')

    if(model_name == 'ATCNet'):
        model = models.ATCNet_(
            n_classes = n_classes, in_chans = n_channels, in_samples = in_samples,
            n_windows = 5, attention = 'mha', eegn_F1 = 16, eegn_D = 2,
            eegn_kernelSize = 64, eegn_poolSize = 7, eegn_dropout = 0.3,
            tcn_depth = 2, tcn_kernelSize = 4, tcn_filters = 32, tcn_dropout = 0.3, tcn_activation='elu',
            )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

#%% 🚀 终极 4 阶段消融对抗大考 (动态信噪比)
def run_comparison():
    # 基础配置
    in_samples, n_sub, n_classes = 1125, 9, 4
    classes_labels = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    data_path = './data/'

    # 动态信噪比：地狱级别
    dynamic_snr = (-34, -29.5)

    def get_exp_conf(channels):
        return {
            'name': 'BCI2a', 'n_classes': n_classes, 'cl_labels': classes_labels,
            'n_sub': n_sub, 'n_channels': channels, 'in_samples': in_samples,
            'data_path': data_path, 'isStandard': True, 'LOSO': False
        }

    train_conf = {'batch_size': 64, 'epochs': 500, 'patience': 100, 'lr': 0.001, 'n_train': 5,
                  'LearnCurves': True, 'from_logits': False, 'model': 'ATCNet'}

    # ==========================================================
    # 1️⃣ [实验 1]：22通道宽频 Naive 组 (全方位作弊顶峰)
    # ==========================================================
    print("\n[EXP 1] 22ch + 0.5-100Hz: Naive Benchmark")
    conf1 = get_exp_conf(22)
    path1 = os.getcwd() + "/results_1_22ch_Wide_Raw"
    # 这里 get_data 必须显式传参
    train(conf1, train_conf, path1, use_br_tad=False, add_noise=False, use_14_channels=False, apply_bandpass=False)
    test(getModel(train_conf['model'], conf1), conf1, path1, use_br_tad=False, add_noise=False, use_14_channels=False,
         apply_bandpass=False)

    # ==========================================================
    # 2️⃣ [实验 2]：14通道宽频 (空间消融 - 揭示边缘通道贡献)
    # ==========================================================
    print("\n[EXP 2] 14ch + 0.5-100Hz: Spatial Ablation Only")
    conf2 = get_exp_conf(14)
    path2 = os.getcwd() + "/results_2_14ch_Wide_Raw"
    train(conf2, train_conf, path2, use_br_tad=False, add_noise=False, use_14_channels=True, apply_bandpass=False)
    test(getModel(train_conf['model'], conf2), conf2, path2, use_br_tad=False, add_noise=False, use_14_channels=True,
         apply_bandpass=False)

    # ==========================================================
    # 3️⃣ [实验 3]：14通道带通 (频率消融 - 揭示高频肌肉贡献)
    # ==========================================================
    print("\n[EXP 3] 14ch + 8-35Hz: Spatial + Frequency Ablation")
    conf3 = get_exp_conf(14)
    path3 = os.getcwd() + "/results_3_14ch_Narrow_Raw"
    train(conf3, train_conf, path3, use_br_tad=False, add_noise=False, use_14_channels=True, apply_bandpass=True)
    test(getModel(train_conf['model'], conf3), conf3, path3, use_br_tad=False, add_noise=False, use_14_channels=True,
         apply_bandpass=True)

    # ==========================================================
    # 4️⃣ [实验 4]：14通道带通 + BR-TAD (源空间净化 - 还原基线)
    # ==========================================================
    print("\n[EXP 4] 14ch + 8-35Hz + BR-TAD: Purified Physiological Base")
    conf4 = get_exp_conf(14)
    path4 = os.getcwd() + "/results_4_14ch_Narrow_BRTAD"
    train(conf4, train_conf, path4, use_br_tad=True, add_noise=False, use_14_channels=True, apply_bandpass=True)
    test(getModel(train_conf['model'], conf4), conf4, path4, use_br_tad=True, add_noise=False, use_14_channels=True,
         apply_bandpass=True)

    # ==========================================================
    # 7️⃣ [实验 5]：终极灾难基线 (14ch + 宽频 + 噪声 + 无洗涤)
    # ==========================================================
    print(f"\n🌪️ [EXP 5] 14ch + 0.5-100Hz + NOISE ({dynamic_snr}dB): Total Disaster Baseline")
    conf7 = get_exp_conf(14)
    path7 = os.getcwd() + "/results_5_Noisy_Wide_Raw"
    train(conf7, train_conf, path7, use_br_tad=False, add_noise=True, snr_range=dynamic_snr, use_14_channels=True,
          apply_bandpass=False)
    test(getModel(train_conf['model'], conf7), conf7, path7, use_br_tad=False, add_noise=True, snr_range=dynamic_snr,
         use_14_channels=True, apply_bandpass=False)

    # ==========================================================
    # 8️⃣ [实验 6]：全频段火力全开 (14ch + 宽频 + 噪声 + BR-TAD)
    # ==========================================================
    print(f"\n🌌 [EXP 6] 14ch + 0.5-100Hz + BR-TAD + NOISE ({dynamic_snr}dB): Full-Band Unleashed")
    conf8 = get_exp_conf(14)
    path8 = os.getcwd() + "/results_6_Noisy_Wide_BRTAD"
    train(conf8, train_conf, path8, use_br_tad=True, add_noise=True, snr_range=dynamic_snr, use_14_channels=True,
          apply_bandpass=False)
    test(getModel(train_conf['model'], conf8), conf8, path8, use_br_tad=True, add_noise=True, snr_range=dynamic_snr,
         use_14_channels=True, apply_bandpass=False)

    # ==========================================================
    # 5️⃣ [实验 7]：地狱压力测试基线 (展示系统彻底崩溃)
    # ==========================================================
    print(f"\n📉 [EXP 7] 14ch + 8-35Hz + NOISE ({dynamic_snr}dB): Baseline Collapse")
    conf5 = get_exp_conf(14)
    path5 = os.getcwd() + "/results_7_Noisy_Raw"
    train(conf5, train_conf, path5, use_br_tad=False, add_noise=True, snr_range=dynamic_snr, use_14_channels=True,
          apply_bandpass=True)
    test(getModel(train_conf['model'], conf5), conf5, path5, use_br_tad=False, add_noise=True, snr_range=dynamic_snr,
         use_14_channels=True, apply_bandpass=True)

    # ==========================================================
    # 5️⃣ [实验 8]：地狱压力测试 (最终系统性能)
    # ==========================================================
    print(f"\n[EXP 8] 14ch + 8-35Hz + BR-TAD + NOISE ({dynamic_snr}dB)")
    conf5 = get_exp_conf(14)
    path5 = os.getcwd() + "/results_8_Noisy_BRTAD"
    train(conf5, train_conf, path5, use_br_tad=True, add_noise=True, snr_range=dynamic_snr, use_14_channels=True,
          apply_bandpass=True)
    test(getModel(train_conf['model'], conf5), conf5, path5, use_br_tad=True, add_noise=True, snr_range=dynamic_snr,
         use_14_channels=True, apply_bandpass=True)
#%%
if __name__ == "__main__":
    run_comparison()