# import random
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import librosa
import matplotlib

# matplotlib.rcParams['figure.dpi'] = 150

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    im.set_clim(-80, 0)   # 控制colorbar的上下限
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class Logger(SummaryWriter):
    
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def log_training(self, training_loss, validation_loss, iteration):
        self.add_scalar("training.loss", training_loss, iteration)
        self.add_scalar("validation.loss", validation_loss, iteration)

    def log(self, loss_term, loss, iteration):
        self.add_scalar(loss_term, loss, iteration)
    
    def log_img(self, gt_spec, predicted_spec, iteration, source):
        D_gt = librosa.amplitude_to_db(gt_spec)
        D_predicted = librosa.amplitude_to_db(predicted_spec)
        self.add_image(
            "gt_%s" % source,
            plot_spectrogram_to_numpy(D_gt),
            iteration, dataformats='HWC')
        self.add_image(
            "predicted_%s" % source,
            plot_spectrogram_to_numpy(D_predicted),
            iteration, dataformats='HWC')

    # def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
    #                  iteration):
    #         self.add_scalar("training.loss", reduced_loss, iteration)
    #         self.add_scalar("grad.norm", grad_norm, iteration)
    #         self.add_scalar("learning.rate", learning_rate, iteration)
    #         self.add_scalar("duration", duration, iteration)

    # def log_validation(self, reduced_loss, model, y, y_pred, iteration):
    #     self.add_scalar("validation.loss", reduced_loss, iteration)
    #     _, mel_outputs, gate_outputs, alignments = y_pred
    #     mel_targets, gate_targets = y

    #     # plot distribution of parameters
    #     for tag, value in model.named_parameters():
    #         tag = tag.replace('.', '/')
    #         self.add_histogram(tag, value.data.cpu().numpy(), iteration)

    #     # plot alignment, mel target and predicted, gate target and predicted
    #     idx = random.randint(0, alignments.size(0) - 1)
    #     self.add_image(
    #         "alignment",
    #         plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
    #         iteration, dataformats='HWC')
    #     self.add_image(
    #         "mel_target",
    #         plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
    #         iteration, dataformats='HWC')
    #     self.add_image(
    #         "mel_predicted",
    #         plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
    #         iteration, dataformats='HWC')
    #     self.add_image(
    #         "gate",
    #         plot_gate_outputs_to_numpy(
    #             gate_targets[idx].data.cpu().numpy(),
    #             torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
    #         iteration, dataformats='HWC')
