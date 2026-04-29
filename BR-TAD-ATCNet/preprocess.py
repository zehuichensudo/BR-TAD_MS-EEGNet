
import numpy as np
import scipy.io as sio
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import json
from mne.io import read_raw_edf
from dateutil.parser import parse
import glob as glob
from datetime import datetime

#%%
# =====================================================================
# 真实物理容积传导伪迹注入引擎 (为消融实验专门设计)
# =====================================================================
def inject_realistic_artifacts(X, BCI2A_CONFIG, snr_range=(-10, -2), noise_type='EMG'):
    """
    参数:
        X: 原始纯净数据 (Trials, Channels, Time)
        BCI2A_CONFIG: 通道配置字典
        snr_range: 信噪比范围元组，例如 (-10, -2) 表示每次 trial 随机生成此范围内的噪声
        noise_type: 'EMG' (高频肌肉毛刺)
    """
    X_noisy = np.copy(X)
    edge_idx = [BCI2A_CONFIG['all_channels'].index(ch) for ch in BCI2A_CONFIG['edge_channels']]
    center_idx = [BCI2A_CONFIG['all_channels'].index(ch) for ch in BCI2A_CONFIG['center_channels']]

    num_trials, num_chans, num_time = X.shape

    for i in range(num_trials):
        trial = X[i]

        # 随机抽取当前 Trial 的真实信噪比
        current_snr_db = np.random.uniform(snr_range[0], snr_range[1])

        # 1. 制造“污染源”物理波形 (EMG: 高频宽带噪声)
        if noise_type == 'EMG':
            noise_source = np.random.randn(num_time)
        else:
            noise_source = np.random.randn(num_time)

        # 2. 严格按当前动态 dB 计算信噪比并缩放噪声
        signal_power = np.var(trial)
        noise_power = np.var(noise_source)
        if noise_power == 0: continue

        target_noise_power = signal_power / (10 ** (current_snr_db / 10.0))
        scale_factor = np.sqrt(target_noise_power / noise_power)
        scaled_base_noise = noise_source * scale_factor

        # 3. 空间容积传导衰减模拟 (边缘向内传导)
        for ch in range(num_chans):
            if ch in edge_idx:
                # 哨兵电极作为污染源，承受 80%~120% 的噪声直击
                spatial_weight = np.random.uniform(0.8, 1.2)
            else:
                # 核心电极在头顶，受颅骨和距离衰减，只承受 10%~40% 的传导
                spatial_weight = np.random.uniform(0.1, 0.4)

            X_noisy[i, ch, :] += scaled_base_noise * spatial_weight

    return X_noisy

#%%
def load_data_LOSO (data_path, subject, dataset):
    X_train, y_train = [], []
    for sub in range (0,9):
        path = data_path+'s' + str(sub+1) + '/'

        if (dataset == 'BCI2a'):
            X1, y1 = load_BCI2a_data(path, sub+1, True)
            X2, y2 = load_BCI2a_data(path, sub+1, False)
        elif (dataset == 'CS2R'):
            X1, y1, _, _, _  = load_CS2R_data_v2(path, sub, True)
            X2, y2, _, _, _  = load_CS2R_data_v2(path, sub, False)

        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        if (sub == subject):
            X_test = X
            y_test = y
        elif len(X_train) == 0:
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


#%%
def load_BCI2a_data(data_path, subject, training, all_trials = True):
    n_channels = 22
    n_tests = 6*48
    window_Length = 7*250
    fs = 250
    t1 = int(1.5*fs)
    t2 = int(6*fs)

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path+'A0'+str(subject)+'T.mat')
    else:
        a = sio.loadmat(data_path+'A0'+str(subject)+'E.mat')
    a_data = a['data']
    for ii in range(0,a_data.size):
        a_data1 = a_data[0,ii]
        a_data2= [a_data1[0,0]]
        a_data3= a_data2[0]
        a_X         = a_data3[0]
        a_trial     = a_data3[1]
        a_y         = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0,a_trial.size):
             if(a_artifacts[trial] != 0 and not all_trials):
                 continue
             data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+window_Length),:22])
             class_return[NO_valid_trial] = int(a_y[trial])
             NO_valid_trial +=1

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return-1).astype(int)

    return data_return, class_return



#%%
def load_CS2R_data_v2(data_path, subject, training,
                      classes_labels =  ['Fingers', 'Wrist','Elbow','Rest'],
                      all_trials = True):
    subjectFiles = glob.glob(data_path + 'S_*/')
    subjectNo = list(dict.fromkeys(sorted([x[len(x)-4:len(x)-1] for x in subjectFiles])))
    if training:  session = 1
    else:         session = 2
    num_runs = 5
    sfreq = 250
    mi_duration = 4.5
    data = np.zeros([num_runs*51, 32, int(mi_duration*sfreq)])
    classes = np.zeros(num_runs * 51)
    valid_trails = 0
    onset = np.zeros([num_runs, 51])
    duration = np.zeros([num_runs, 51])
    description = np.zeros([num_runs, 51])

    CheckFiles = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/*.edf')
    if not CheckFiles:
        return

    for runNo in range(num_runs):
        valid_trails_in_run = 0
        EDFfile = glob.glob(data_path + 'S_' + subjectNo[subject].zfill(3) + '/S' + str(session) + '/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.edf')
        JSONfile = glob.glob(data_path + 'S_'+subjectNo[subject].zfill(3) + '/S'+ str(session) +'/S_'+subjectNo[subject].zfill(3)+'_'+str(session)+str(runNo+1)+'*.json')

        if not EDFfile: continue
        raw = read_raw_edf(str(EDFfile[0]), preload=True, verbose=False)
        f = open(JSONfile[0],); JSON = json.load(f)
        keyStrokes = np.min([len(JSON['Markers']), 51])
        date_string = EDFfile[0][-21:-4]
        datetime_format = "%d.%m.%y_%H.%M.%S"
        startRecordTime = datetime.strptime(date_string, datetime_format).astimezone()

        currentTrialNo = 0
        if(runNo == 4): currentTrialNo = 4

        ch_names = raw.info['ch_names'][4:36]
        raw.filter(4., 50., fir_design='firwin')
        raw = raw.copy().pick_channels(ch_names = ch_names)
        raw = raw.copy().resample(sfreq = sfreq)
        fs = raw.info['sfreq']

        for trail in range(keyStrokes):
            if(runNo == 4 ): currentTrialNo = 4
            elif (currentTrialNo == 3): currentTrialNo = 1
            else: currentTrialNo = currentTrialNo + 1

            trailDuration = 8
            trailTime = parse(JSON['Markers'][trail]['startDatetime'])
            trailStart = trailTime - startRecordTime
            trailStart = trailStart.seconds
            start = trailStart + (6 - mi_duration)
            stop = trailStart + 6

            if (trail < keyStrokes-1):
                trailDuration = parse(JSON['Markers'][trail+1]['startDatetime']) - parse(JSON['Markers'][trail]['startDatetime'])
                trailDuration =  trailDuration.seconds + (trailDuration.microseconds/1000000)
                if (trailDuration < 7.5) or (trailDuration > 8.5):
                    if (trailDuration > 14 and trailDuration < 18):
                        if (currentTrialNo == 3):   currentTrialNo = 1
                        else:                       currentTrialNo = currentTrialNo + 1
                    continue
            elif (trail == keyStrokes-1):
                trailDuration = raw[0, int(trailStart*int(fs)):int((trailStart+8)*int(fs))][0].shape[1]/fs
                if (trailDuration < 7.8) : continue

            MITrail = raw[:32, int(start*int(fs)):int(stop*int(fs))][0]
            if (MITrail.shape[1] != data.shape[2]): return

            if ((('Fingers' in classes_labels) and (currentTrialNo==1)) or
            (('Wrist' in classes_labels) and (currentTrialNo==2)) or
            (('Elbow' in classes_labels) and (currentTrialNo==3)) or
            (('Rest' in classes_labels) and (currentTrialNo==4))):
                data[valid_trails] = MITrail
                classes[valid_trails] =  currentTrialNo
                onset[runNo, valid_trails_in_run]  = start
                duration[runNo, valid_trails_in_run] = trailDuration - (6 - mi_duration)
                description[runNo, valid_trails_in_run] = currentTrialNo
                valid_trails += 1
                valid_trails_in_run += 1

    data = data[0:valid_trails, :, :]
    classes = classes[0:valid_trails]
    classes = (classes-1).astype(int)

    return data, classes, onset, duration, description


#%%
def standardize_data(X_train, X_test, channels):
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, 0, j, :])
          X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
          X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


def get_data(path, subject, dataset='BCI2a', classes_labels='all', LOSO=False, isStandard=True, isShuffle=True,
             use_14_channels=True,
             apply_bandpass=True,
             use_br_tad=False, add_noise=False, snr_range=(-10, -2)):
    if LOSO:
        X_train, y_train, X_test, y_test = load_data_LOSO(path, subject, dataset)
    else:
        if (dataset == 'BCI2a'):
            path_sub = path + 's{:}/'.format(subject + 1)
            X_train, y_train = load_BCI2a_data(path_sub, subject + 1, True)
            X_test, y_test = load_BCI2a_data(path_sub, subject + 1, False)

            from br_tad import BCI2A_CONFIG

            # 注入噪声
            if add_noise:
                print(f"💣 正在向 Subject {subject + 1} 注入动态范围 {snr_range}dB 的容积传导 EMG 干扰...")
                X_train = inject_realistic_artifacts(X_train, BCI2A_CONFIG, snr_range=snr_range, noise_type='EMG')
                X_test = inject_realistic_artifacts(X_test, BCI2A_CONFIG, snr_range=snr_range, noise_type='EMG')

            # 空间消融与洗涤
            center_indices = [BCI2A_CONFIG['all_channels'].index(ch) for ch in BCI2A_CONFIG['center_channels']]

            if use_br_tad:
                from br_tad import Orthogonal_Source_BR_TAD_Engine
                print(f"🌀 正在使用 BR-TAD 洗涤 Subject {subject + 1} 的数据...")
                engine = Orthogonal_Source_BR_TAD_Engine(BCI2A_CONFIG)

                _, X_train_clean, _ = engine.process_all(X_train)
                _, X_test_clean, _ = engine.process_all(X_test)

                if use_14_channels:
                    X_train = X_train_clean[:, center_indices, :]
                    X_test = X_test_clean[:, center_indices, :]
                    print(f"✅ 洗涤与降维完成！当前通道数: {X_train.shape[1]}")
                else:
                    X_train = X_train_clean
                    X_test = X_test_clean
                    print(f"✅ 洗涤完成！保持全通道: {X_train.shape[1]}")
            else:
                if use_14_channels:
                    X_train = X_train[:, center_indices, :]
                    X_test = X_test[:, center_indices, :]
                # 如果 use_14_channels 为 False，则自动保留 22 通道

            #  频率消融
            from scipy.signal import butter, filtfilt
            def apply_filter(data, low, high):
                b, a = butter(4, [low / 125.0, high / 125.0], btype='band')
                return filtfilt(b, a, data, axis=-1).astype(np.float32)

            if apply_bandpass:
                print("🧠 [物理锁] 强制锁定 8-35Hz 核心防区...")
                X_train = apply_filter(X_train, 8, 35)
                X_test = apply_filter(X_test, 8, 35)
            else:
                print("🔓 [宽频态] 保持 0.5-100Hz 原始状态...")
                X_train = apply_filter(X_train, 0.5, 100)
                X_test = apply_filter(X_test, 0.5, 100)

        elif (dataset == 'CS2R'):
            X_train, y_train, _, _, _ = load_CS2R_data_v2(path, subject, True, classes_labels)
            X_test, y_test, _, _, _ = load_CS2R_data_v2(path, subject, False, classes_labels)
        else:
            raise Exception("暂不支持 '{}' 数据集!".format(dataset))

    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)

    N_ts, _, _ = X_test.shape
    X_test = X_test.reshape(N_ts, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)

    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot