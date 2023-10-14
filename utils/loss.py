# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        print ("torch.sigmoid(pred): ",pred)

        dx = pred - true  # reduce only missing label effects
        print("dx: ", dx)

        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        print("alpha_factor: ", alpha_factor)

        loss *= alpha_factor
        print("loss *= alpha_factor: ", loss)

        print("loss.mean(): ", loss.mean())
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        print("device: ",device)

        h = model.hyp  # hyperparameters
        print ("h_compute_loss: ",h )

        # Define criteria
        print ("h['cls_pw']: ",h['cls_pw'])
        print("torch.tensor([h['cls_pw']]: ", torch.tensor([h['cls_pw']]))
        print("h['obj_pw']: ",h['obj_pw'])
        print("torch.tensor([h['obj_pw']]: ",torch.tensor([h['obj_pw']]))

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        print("BCEcls: ",BCEcls)
        print("BCEobj: ",BCEobj)
        print("eps: ",h.get("label_smoothing",0.0))
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        print("self.cp: ",self.cp,"self.cn: ",self.cn)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        print("h['fl_gamma'] === g: ",g)
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        print ("model.gr: ",model.gr)
        print ("is_parallel(model): ",is_parallel(model))
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        print("det:\n", det)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        print ("self.balance: ",self.balance)
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        print("self.ssi : ", self.ssi)

        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            print("getattr(det, k): ", getattr(det, k))
            setattr(self, k, getattr(det, k))
        print ("self.anchors: ",self.anchors)
        print("self.na: ", self.na)
        print("self.nc: ", self.nc)
        print("self.nl: ", self.nl)

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        #print ("p: ",p)
        for i, pi in enumerate(p):  # layer index, layer predictions

            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            print ("b-image: \n",b)
            print("a-anchor: \n", a)
            print("gj-gridy: \n", gj)
            print("gi-gridx: \n", gi)

            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            print ("tobj: ",tobj)
            print("tobj.size: ", tobj.size())

            n = b.shape[0]  # number of targets
            print("number of targets: ", n)
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                print ("ps::: ",ps)

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                print ("pxy-regression: \n",pxy)
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                print("pwh-regression: \n", pwh)
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                print("pbox-regression: \n", pbox)

                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                print("iou-regression: \n", iou)

                lbox += (1.0 - iou).mean()  # iou loss
                print("lbox-regression: \n", lbox)

                # Objectness
                print ("self.gr: ",self.gr)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                print("tobj-Objectness:::: ",tobj)
                print("tobj-Objectness.size():::: ", tobj.size())
                print("torch>0: ",torch.where(tobj >0))

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    print ("Classification:::::: ")
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    print ("n:^: ",n)
                    print ("t:^: ",t)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):

        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        print("number of anchors: ", na)
        print("number of targets: ", nt)

        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        print ("gain: ",gain)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        print ("ai: ",ai)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        print ("targets_l: ",targets)
        print("targets_l.size: ", targets.size())

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        print ("off: ",off)
        print ( "self.nl_in loss: ",self.nl)
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            print ("gain[2:6]-transf: ",gain)

            # Match targets to anchors
            t = targets * gain
            print ("targets * gain: \n",t)
            if nt:
                # Matches
                print ("t[:, :, 4:6]: ",t[:, :, 4:6])
                print("anchors[:, None]: ", anchors[:, None])

                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                print ("r - matches: ",r)
                print ("torch.max(r, 1. / r).max(2)[0]: ",torch.max(r, 1. / r).max(2)[0])
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                print("j::: ",j)

                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                print ("t_loss_filter: ",t )

                # Offsets
                gxy = t[:, 2:4]  # grid xy

                print ("grid xy: ",gxy)
                print ("gain[[2, 3]]: ",gain[[2, 3]])
                gxi = gain[[2, 3]] - gxy  # inverse
                print ("inverse grid xy: ",gxi)
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                print ("j::: ",j)
                print("k::: ", k)
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                print("l::: ", l)
                print("m::: ", m)

                j = torch.stack((torch.ones_like(j),))
                print ("torch.stack((torch.ones_like(j),)): ",j)
                print ("t.size_1: ",j)
                t = t.repeat((off.shape[0], 1, 1))[j]
                print("t.repeat((off.shape[0], 1, 1))[j]: ", t)

                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                print ("ofset::: ",offsets)
                print("ofset.size::: ", offsets.size())
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            print ("b:::: ",b)
            print("c:::: ", c)
            gxy = t[:, 2:4]  # grid xy
            print('gxy:::z ',gxy)
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            print("gij::: ",gij)

            gi, gj = gij.T  # grid xy indices
            print ("gi:::z ",gi)
            print("gj::: ", gj)

            # Append
            a = t[:, 6].long()  # anchor indices
            #print ("anchor indices-a: ",a)
            #print ("gj.clamp_(0, gain[3] - 1): ",gj.clamp_(0, gain[3] - 1))
            #print ("gi.clamp_(0, gain[2] - 1)): ",gi.clamp_(0, gain[2] - 1))
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            #print ("tbox: ",tbox)
            anch.append(anchors[a])  # anchors
            #print ("anch: ",anch)
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
