import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    loss_history = []
    max_auc = 0

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_s.to('cuda')
        data_a.to('cuda')

    # Train model
    lbl = data_a.y
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    for epoch in range(args.epochs):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            output, cla_os, cla_os_a, _ = model(data_o, data_s, data_a, inp)

            log = torch.squeeze(m(output))
            loss1 = loss_fct(log, label.float())
            loss2 = b_xent(cla_os, lbl.float())
            loss3 = b_xent(cla_os_a, lbl.float())
            loss_train = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

            loss_history.append(loss_train)
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation after each epoch
        if not args.fastmode:
            roc_val, prc_val, f1_val, loss_val = test(model, val_loader, data_o, data_s, data_a, args)
            if roc_val > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = roc_val

            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(roc_train),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'auroc_val: {:.4f}'.format(roc_val),
                  'auprc_val: {:.4f}'.format(prc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'time: {:.4f}s'.format(time.time() - t))
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    plt.plot(loss_history)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test = test(model_max, test_loader, data_o, data_s, data_a, args)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))

    with open(args.out_file, 'a') as f:
        f.write('{0}\t{1}\t{2}\t{7}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\n'.format(
            args.in_file[5:8], args.seed, args.aggregator, loss_test.item(), auroc_test, prc_test, f1_test, args.feature_type))


def test(model, loader, data_o, data_s, data_a, args):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    lbl = data_a.y

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            if args.cuda:
                label = label.cuda()

            output, cla_os, cla_os_a, _ = model(data_o, data_s, data_a, inp)
            log = torch.squeeze(m(output))

            loss1 = loss_fct(log, label.float())
            loss2 = b_xent(cla_os, lbl.float())
            loss3 = b_xent(cla_os_a, lbl.float())
            loss = args.loss_ratio1 * loss1 + args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss