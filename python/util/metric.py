import torch
import numpy as np
import pdb

def nmse_metric_np(input, trainingMask, target, th=0.5, needSigmoid=True):
    '''
    validate the 2D CNN bases model 
    '''
    if needSigmoid:
        input = torch.sigmoid(input)
        trainingMask = torch.sigmoid(trainingMask)
    trainingMask[trainingMask>=th] = 1
    trainingMask[trainingMask<th] = 0
    input = input * trainingMask
    res = input-target
    res = res.data.cpu().numpy()
    target = target.data.cpu().numpy()
    res_c = res[:,0,:,:] + 1j*res[:,1,:,:]
    target_c = target[:,0,:,:] + 1j*target[:,1,:,:]
    res_c_norm = np.sum(abs(res_c)**2, axis=(1,2))
    target_c_norm = np.sum(abs(target_c)**2, axis=(1,2))
    nmse = res_c_norm / target_c_norm
    return np.mean(nmse)

def nmse_metirc_lista(input, target):
    '''
    validate the lista model, batch size at axis 1 not axis 0
    '''
    res = input - target
    res = res.data.cpu().numpy()
    target = target.data.cpu().numpy()
    s, b = target.shape
    s = int(s/2)
    res_c = res[:s, :] + 1j*res[s:, :]
    target_c = target[:s, :] + 1j*target[s:, :]
    res_c_norm = np.sum(abs(res_c)**2, axis=0)
    target_c_norm = np.sum(abs(target_c)**2, axis=0)
    mask = target_c_norm != 0
    nmse = res_c_norm[mask] / target_c_norm[mask]
    return np.mean(nmse)


class runningScore(object):
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, target, pred, n_classes):
        mask = (target>=0) & (target < n_classes)
        if np.sum((pred[mask]<0)) > 0:
            print(pred[pred<0])
        hist = np.bincount(
            n_classes * target[mask].astype(int) + pred[mask],
            minlength=n_classes**2).reshape(n_classes, n_classes)
        return hist
    def update(self, targets, preds, reverseLabel=False):
        '''
            reverseLabel only works when n_classes=2
        '''
        if self.n_classes==2 and reverseLabel:
            targets = 1 - targets
            preds = 1 - preds
        for target, pred in zip(targets, preds):
            self.confusion_matrix += self._fast_hist(target.flatten(), pred.flatten(), self.n_classes)
    
    def get_scores(self):
        epson = 1e-6
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() /( hist.sum() + epson)
        if self.n_classes==2:
            tp = self.confusion_matrix[1,1]
            fn = self.confusion_matrix[1,0]
            fp = self.confusion_matrix[0,1]
            precision = tp / (tp + fp + epson)
            recall = tp / (tp + fn + epson)
            f1_score = 2 * precision * recall / (precision + recall)
            return {"acc": acc,
                    "recall": recall,
                    "precision": precision,
                    "f1_score": f1_score}
        else:
            raise RuntimeError('Not implement yet')
        '''
        fn = np.zeros(self.n_classes)
        fp = np.zeros(self.n_classes)
        for class_i in range(self.n_classes):
            for class_j in range(self.n_classes):
                if class_i != class_j:
                    fn[class_i] += hist[class_i,class_j]
                    fp[class_i] += hist[class_j,class_i]
                
        recall = tp / (tp + fn + epson)
        #recall = np.nanmean(recall)
        precision = tp / (tp + fp + epson)
        #precision = np.nanmean(precision)
        f1_score = 2 * precision * recall / (precision + recall)
        return {'Overall Acc': acc,
                'Recall':recall,
                'Precision':precision,
                'F1':f1_score}
        '''

    def reset(self):
        self.confusion_matrix = np.zeros(self.n_classes, self.n_classes)



