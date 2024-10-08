import os, inspect, time, math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+160, (x*dw):(x*dw)+64, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot_3d_and_2d(contents, savename=""):
    fig = plt.figure(figsize=(12, 6))

    # 3D 산점도
    ax1 = fig.add_subplot(121, projection='3d')
    for i, content in enumerate(contents):
        z = content
        x = np.full_like(z, i)  # X축을 데이터의 인덱스로 설정
        y = np.arange(len(z))   # Y축을 데이터 포인트의 인덱스로 설정
        ax1.scatter(x, y, z, label=f'class-{i+1}')

    ax1.set_xlabel('Class')
    ax1.set_ylabel('Sample Index')
    ax1.set_zlabel('Value')
    ax1.legend()

    # 2D 플롯
    ax2 = fig.add_subplot(122)
    for i, content in enumerate(contents):
        y = content
        x = np.arange(len(y))  # Sample Index를 X축으로 설정
        ax2.plot(x, y, label=f'class-{i+1}', marker='o', linestyle='none')

    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Value')
    ax2.legend()

    plt.tight_layout()
    
    # 2D 플롯 저장
    if savename:
        plt.savefig(savename)

    plt.show()
    plt.close()

def histogram(contents, savename=""):

    n1, _, _ = plt.hist(contents[0], bins=100, alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(contents[1], bins=100, alpha=0.5, label='Abnormal')
    h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(contents[0].max(), contents[1].max())
    plt.xlim(0, xmax)
    plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    plt.savefig(savename)
    plt.close()

def training(neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="results")
    result_list = ["tr_resotring"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    start_time = time.time()
    iteration = 0

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch

        x_restore, _, _, _ = neuralnet.step(x=x_tr, iteration=iteration, train=False)

        save_img(contents=[x_tr, x_restore, (x_tr-x_restore)**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.
            _, mse, w_etrp, loss = neuralnet.step(x=x_tr, iteration=iteration, train=True)
            neuralnet.save_params()

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  MSE:%.3f, W-ETRP:%.3f, Total:%.3f" \
            %(epoch, epochs, iteration, np.sum(mse), np.sum(w_etrp), loss))

def test(neuralnet, dataset, batch_size):

    neuralnet.load_params()

    print("\nTest...")

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    scores_normal, scores_abnormal = [], []
    
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)

        _, score_anomaly, _, _ = neuralnet.step(x=x_te, iteration=0, train=False)
        if y_te == 1: 
            scores_normal.append(score_anomaly)
        else: 
            scores_abnormal.append(score_anomaly)

        if terminator: break

    # Calculate statistics after processing all test data
    scores_normal = np.asarray(scores_normal)
    scores_abnormal = np.asarray(scores_abnormal)
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    abnormal_avg, abnormal_std = np.average(scores_abnormal), np.std(scores_abnormal)
    outbound = normal_avg + (normal_std * 3)

    print("Normal  avg: %.5f, std: %.5f" %(normal_avg, normal_std))
    print("Abnormal  avg: %.5f, std: %.5f" %(abnormal_avg, abnormal_std))
    print("Outlier boundary of normal data: %.5f" %(outbound))

    # Initialize metrics
    tp, fp, fn, tn = 0, 0, 0, 0  # True Positives, False Positives, False Negatives, True Negatives
    
    for score_anomaly, y_te in zip(np.concatenate([scores_normal, scores_abnormal]), [1] * len(scores_normal) + [0] * len(scores_abnormal)):
        if score_anomaly > outbound:
            if y_te == 0:  # 실제 비정상
                tp += 1  # True Positive
            else:  # 실제 정상
                fp += 1  # False Positive
        else:
            if y_te == 0:  # 실제 비정상
                fn += 1  # False Negative
            else:  # 실제 정상
                tn += 1  # True Negative

    # Calculate Precision, Recall, Accuracy, F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"tp: {tp:f}")
    print(f"fp: {fp:f}")
    print(f"tn: {tn:f}")
    print(f"fn: {fn:f}")
    

    histogram(contents=[scores_normal, scores_abnormal], savename="histogram-test.png")

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.

        x_restore, restore_loss, _, _ = neuralnet.step(x=x_te, iteration=0, train=False)

        loss4box[y_te[0]].append(restore_loss)

        outcheck = restore_loss > outbound
        fcsv.write("%d, %.3f, %r\n" %(y_te, restore_loss, outcheck))

        [h, w, c] = x_te[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = x_restore[0]
        canvas[:, w*2:, :] = (x_te[0]-x_restore[0])**2
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(restore_loss))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(restore_loss))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    # 3D 박스플롯 함수 호출
    boxplot_3d_and_2d(contents=loss4box, savename="test-box.png")
