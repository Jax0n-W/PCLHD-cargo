from __future__ import absolute_import, print_function

import time

import torch

from .trainers import ClusterContrastTrainer_PCLMP
from .utils.meters import AverageMeter


def compose_primary_loss_without_ema(loss_ir, loss_rgb, cross_loss):
    """Compose the first Stage 2 optimization loss without ``loss_ema``."""
    return loss_ir + loss_rgb + 0.25 * cross_loss


class ClusterContrastTrainer_PCLMP_NoEMALoss(ClusterContrastTrainer_PCLMP):
    """Stage 2 ablation that excludes only EMA loss from optimization.

    The EMA forward pass, EMA-memory monitoring, second ALL-memory optimizer
    step, EMA parameter update, and the caller's EMA evaluation/checkpoint path
    intentionally remain identical to the original trainer.
    """

    def train(self, epoch, data_loader_ir, data_loader_rgb, data_loader_all_ir,
              data_loader_all_rgb, optimizer, print_freq=10, train_iters=400,
              i2r=None, r2i=None):
        self.encoder.train()
        self.encoder_ema.train()

        print("============================================")
        print("ABLATION: NO EMA LOSS ONLY")
        print("Trainer source: {}".format(__file__))
        print("EMA loss included: False")
        print("ALL memory optimization enabled: True")
        print("============================================")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = \
                self._parse_data_rgb(inputs_rgb)

            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir = \
                self._forward(inputs_rgb, inputs_ir, label_1=labels_rgb,
                              label_2=labels_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            if r2i:
                rgb2ir_labels = torch.tensor(
                    [r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor(
                    [i2r[key.item()] for key in labels_ir]).cuda()
                alternate = True
                if alternate:
                    if epoch % 2 == 1:
                        cross_loss = self.memory_rgb(
                            f_out_ir, ir2rgb_labels.long())
                    else:
                        cross_loss = self.memory_ir(
                            f_out_rgb, rgb2ir_labels.long())
                else:
                    cross_loss = (
                        self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                        + self.memory_ir(f_out_rgb, rgb2ir_labels.long()))
            else:
                cross_loss = loss_ir.new_zeros(())

            # Keep the original EMA forward and monitoring values. Only their
            # contribution to the optimized loss is removed in this ablation.
            with torch.no_grad():
                (_, f_out_rgb_ema, f_out_ir_ema, labels_rgb_ema,
                 labels_ir_ema, pool_rgb_ema, pool_ir_ema) = self._forward_ema(
                    inputs_rgb, inputs_ir, label_1=labels_rgb,
                    label_2=labels_ir, modal=0)
            loss_ir_ema = self.memory_ir(
                f_out_ir_ema, labels_ir_ema, model_name="encoder_ema")
            loss_rgb_ema = self.memory_rgb(
                f_out_rgb_ema, labels_rgb_ema, model_name="encoder_ema")
            loss_ema = loss_ir_ema + loss_rgb_ema

            loss = compose_primary_loss_without_ema(
                loss_ir, loss_rgb, cross_loss)

            if i == 0:
                print("No EMA Loss ablation: loss_ema is monitored but "
                      "excluded from the optimized Stage 2 loss "
                      "(requires_grad={}).".format(loss_ema.requires_grad))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            inputs_all_ir = data_loader_all_ir.next()
            inputs_all_rgb = data_loader_all_rgb.next()
            inputs_all_ir, labels_all_ir, indexes_all_ir = \
                self._parse_data_ir(inputs_all_ir)
            inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = \
                self._parse_data_rgb(inputs_all_rgb)

            inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
            labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)
            (_, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir,
             pool_all_rgb, pool_all_ir) = self._forward(
                inputs_all_rgb, inputs_all_ir, label_1=labels_all_rgb,
                label_2=labels_all_ir, modal=0)

            loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
            loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)
            loss2 = loss_all_ir + loss_all_rgb

            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Optimized loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'EMA monitor (excluded) {:.3f}\t'
                      'Loss ir ema {:.3f}\t'
                      'Loss rgb ema {:.3f}\t'
                      'Loss all {:.3f}\t'
                      'Loss all ir {:.3f}\t'
                      'Loss all rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb,
                              loss_ema, loss_ir_ema, loss_rgb_ema, loss2,
                              loss_all_ir, loss_all_rgb))
