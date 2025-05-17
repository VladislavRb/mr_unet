import torch
import torch.nn as nn


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes: list[int] = [256, 512, 1024],
                 hop_sizes: list[int] = [64, 128, 256],
                 win_lengths: list[int] = [256, 512, 1024]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_complex = pred[:, 0] + 1j * pred[:, 1]
        target_complex = target[:, 0] + 1j * target[:, 1]

        pred_complex = pred_complex.permute(0, 2, 1)
        target_complex = target_complex.permute(0, 2, 1)

        pred_wav = torch.istft(pred_complex, n_fft=1024, hop_length=512, length=target.shape[-1])
        target_wav = torch.istft(target_complex, n_fft=1024, hop_length=512, length=target.shape[-1])

        loss = 0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_stft = MultiResolutionSTFTLoss._stft(pred_wav, fft_size, hop_size, win_length)
            target_stft = MultiResolutionSTFTLoss._stft(target_wav, fft_size, hop_size, win_length)
            loss += self.l1(torch.abs(pred_stft), torch.abs(target_stft))

        return loss

    @staticmethod
    def _stft(x, fft_size, hop_size, win_length):
        return torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, return_complex=True)
