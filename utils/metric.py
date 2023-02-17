#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'


from sklearn import metrics
import torch
import torch.nn.functional as F
import numpy as np
from utils import helper
from tqdm import tqdm
args = helper.get_args()


def test_batch(model, data, device, dtype, required_feats=False):
    out_list = []
    model.to(device)
    with torch.no_grad():
        if dtype == "image":
            x, y = data[0].to(device), data[1].to(device)
            if required_feats:
                logits, out_list = model.feature_list(x)
            else:
                logits = model(x)
            loss = F.cross_entropy(logits, y).item()
            pred = logits.argmax(dim=1).detach().cpu()
        elif dtype == "graph":
            for i in range(len(data)):
                data[i] = data[i].to(device)
            x, A, mask, max_nodes, y = data

            if required_feats:
                logits, out_list = model.feature_list(data)
            else:
                logits = model(data)
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            loss = F.cross_entropy(logits, data[4]).item()
            pred = logits.argmax(dim=1).detach().cpu()
        elif dtype == "nlp":
            pass
        else:
            raise NotImplementedError(f"-> dtype:{dtype} not implemented!!")
        top1 = top3 = top5 = torch.eq(y.cpu().view_as(pred), pred).sum().item()
        if logits.shape[1] >= 3:
            _, tk = torch.topk(logits, k=3, dim=1)
            top3 = torch.eq(y[:, None, ...], tk).any(dim=1).sum().item()
        if logits.shape[1] >= 5:
            _, tk = torch.topk(logits, k=5, dim=1)
            top5 = torch.eq(y[:, None, ...], tk).any(dim=1).sum().item()
        return logits, pred, top1, top3, top5, out_list, loss

_best_topk_acc = {
    "top1": 0,
    "top3": 0,
    "top5": 0,
}
def topk_test(model, test_loader, device, epoch=0, debug=False):
    global _best_topk_acc
    test_loss = 0.0
    correct = {
        "top1": 0,
        "top3": 0,
        "top5": 0,
    }
    topk_acc = {
        "top1": 0,
        "top3": 0,
        "top5": 0,
    }
    model.to(device)
    size = 0

    for data in test_loader:
        _outs, _pred, _top1, _top3, _top5, _out_list, _loss = \
            test_batch(model, data, device, dtype=test_loader.dtype, required_feats=False)
        test_loss += _loss
        correct["top1"] += _top1
        correct["top3"] += _top3
        correct["top5"] += _top5
        size += _pred.shape[0]

    test_loss /= (1.0 * size)
    topk_acc["top1"] = round(100.0 * correct["top1"] / size, 5)
    topk_acc["top3"] = round(100.0 * correct["top3"] / size, 5)
    topk_acc["top5"] = round(100.0 * correct["top5"] / size, 5)
    for k, v in topk_acc.items():
        if v > _best_topk_acc[k]:
            _best_topk_acc[k] = v
    msg = "-> For E{:d}, [Test] loss={:.5f}, top-1={:.3f}%, top-3={:.3f}%, top-5={:.3f}%".format(
            int(epoch),
            test_loss,
            topk_acc["top1"],
            topk_acc["top3"],
            topk_acc["top5"]
    )
    if debug: print(msg)
    return _best_topk_acc, topk_acc, test_loss


def multi_mertic(y, p, scores=None):
    result_dict = {}
    t_idx = np.where(y == 0)[0]
    f_idx = np.where(y == 1)[0]
    TP = len(np.where(p[t_idx] == 0)[0])
    FP = len(np.where(p[f_idx] == 0)[0])
    FN = len(np.where(p[t_idx] == 1)[0])
    TN = len(np.where(p[f_idx] == 1)[0])
    print(f"-> TP={TP}, FP={FP}, FN={FN}, TN={TN}")

    result_dict["TP"] = TP
    result_dict["FP"] = FP
    result_dict["FN"] = FN
    result_dict["TN"] = TN
    result_dict["FPR100"] = 100.0 * FP / (FP + TN + 1e-6)
    result_dict["TPR100"] = 100.0 * TP / (TP + FN + 1e-6)
    result_dict["ACC"] = round(100.0 * (TP + TN) / (TP + FP + TN + FN + 1e-6), 5)
    result_dict["Recall"] = round(100.0 * (TP) / (TP + FN + 1e-6), 5)
    result_dict["Precision"] = round(100.0 * (TP) / (TP + FP + 1e-6), 5)
    result_dict["F1score"] = round((2.0 * result_dict["Precision"] * result_dict["Recall"]) /
                                   (result_dict["Precision"] + result_dict["Recall"] + 1e-6), 5)

    if scores is not None:
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
        result_dict["FPR"] = fpr
        result_dict["TPR"] = tpr
        result_dict["thresholds"] = thresholds
    return result_dict


def test(classifier, ben_test_loader, adv_test_loader, epoch=0, tau1=0.5, file_path=None, info=""):
    classifier.eval()
    classifier = classifier.to(args.device)
    total, sum_query_correct, sum_correct, sum_loss = 0, 0, 0, 0.0,
    result_dict = {
        "y": [],
        "pred": [],
        "query_y": [],
        "query_pred": [],
        "scores": [],
        "ben_scores": [],
        "adv_scores": [],
        "conf": []
    }

    batch_size = ben_test_loader.batch_size
    with torch.no_grad():
        step = 0
        for (x, y) in ben_test_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            output = classifier(x)
            loss = F.cross_entropy(output, y)
            total += y.size(0)
            sum_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            scores = F.softmax(output)[:, 1].detach().cpu()

            result_dict["scores"].append(scores)
            result_dict["y"].append(y.detach().cpu())
            result_dict["pred"].append(pred.view([-1]).detach().cpu())
            sum_correct += pred.eq(y.view_as(pred)).sum().item()

            query_scores = float(1.0 * pred.sum() / len(pred))
            detect_flag = 1 if query_scores > tau1 else 0
            truth_flag = 1 if (1.0 * y.sum() / len(y)) > tau1 else 0
            result_dict["query_y"].append(truth_flag)
            result_dict["query_pred"].append(detect_flag)
            result_dict["ben_scores"].append(query_scores)
            if detect_flag == truth_flag:
                sum_query_correct += 1
            res_info = "[Test] epoch:{:d} Loss: {:.6f} Acc:{:.3f}%".format(
                epoch,
                sum_loss / total,
                100.0 * sum_correct / total
            )
            helper.progress_bar(step, 2*len(adv_test_loader), res_info)
            step += 1

        for (x, y) in adv_test_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            output = classifier(x)
            loss = F.cross_entropy(output, y)
            total += y.size(0)
            sum_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            scores = F.softmax(output)[:, 1].detach().cpu()
            result_dict["scores"].append(scores)
            result_dict["y"].append(y.detach().cpu())
            result_dict["pred"].append(pred.view([-1]).detach().cpu())
            sum_correct += pred.eq(y.view_as(pred)).sum().item()
            query_scores = float(1.0 * pred.sum() / len(pred))
            detect_flag = 1 if query_scores > tau1 else 0
            truth_flag = 1 if (1.0 * y.sum() / len(y)) > tau1 else 0
            result_dict["query_y"].append(truth_flag)
            result_dict["query_pred"].append(detect_flag)
            result_dict["adv_scores"].append(query_scores)
            if detect_flag == truth_flag:
                sum_query_correct += 1
            res_info = "[Test] epoch:{:d} Loss: {:.6f} Acc:{:.3f}%".format(
                epoch,
                sum_loss / total,
                100.0 * sum_correct / total
            )
            helper.progress_bar(step, 2*len(adv_test_loader), res_info)
            step += 1

    result_dict["y"] = torch.cat(result_dict["y"]).detach().cpu().numpy()
    result_dict["pred"] = torch.cat(result_dict["pred"]).detach().cpu().numpy()
    result_dict["scores"] = torch.cat(result_dict["scores"]).detach().cpu().numpy()
    result_dict["query_y"] = torch.tensor(result_dict["query_y"]).detach().cpu().numpy()
    result_dict["query_pred"] = torch.tensor(result_dict["query_pred"]).detach().cpu().numpy()
    result_dict["ben_scores"] = np.array(result_dict["ben_scores"])
    result_dict["adv_scores"] = np.array(result_dict["adv_scores"])
    result_dict["query_scores"] = np.concatenate([result_dict["ben_scores"], result_dict["adv_scores"]])


    ben_avg = float(result_dict["ben_scores"].sum() / result_dict["ben_scores"].shape[0])
    adv_avg = float(result_dict["adv_scores"].sum() / result_dict["adv_scores"].shape[0])

    y = result_dict["y"]
    p = result_dict["pred"]
    sample_res = multi_mertic(y, p)

    y = result_dict["query_y"]
    p = result_dict["query_pred"]
    query_res = multi_mertic(y, p)

    msg1 =f"""<<<====================================={info}=======================================>>>\n
    -> TEST_SAMPLE(τ1={tau1}, bs=1) ACC:{sample_res['ACC']}% 
        -> Recall:{sample_res["Recall"]} Precision:{sample_res["Precision"]} F1-score:{sample_res["F1score"]}
        -> FPR:{sample_res['FPR100']} TPR:{sample_res['TPR100']}\n"""

    msg2 = f"""
    -> TEST_QUERY(τ1={tau1}, bs={batch_size}) ACC:{query_res['ACC']}% 
        -> Recall:{query_res["Recall"]} Precision:{query_res["Precision"]} F1-score:{query_res["F1score"]}
        -> FPR:{query_res['FPR100']} TPR:{query_res['TPR100']}
        -> Ben_avg_score: {ben_avg} Adv_avg_score:{adv_avg}\n\n\n\n"""

    print(msg1)
    print()
    print(msg2)

    result_dict.update(query_res)
    if file_path is not None:
        #print("-> save result", file_path)
        torch.save(result_dict, file_path)
        with open(file_path[:-3] + ".txt", "a") as fp:
            fp.write(msg1 + msg2 + "\n\n\n")
    return result_dict


def train(net, train_loader, optimizer, epochs=1):
    net.train()
    net.to(args.device)
    phar = tqdm(range(1, 1+epochs))
    for epoch in phar:
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device).view(len(x), -1), y.to(args.device)
            optimizer.zero_grad()
            out = net(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        phar.set_description(f"-> training for epoch:{epoch}")
    return net