import torch
import torch.nn as nn
from torch.nn import LayerNorm, BatchNorm2d
import torchaudio.compliance.kaldi as ta_kaldi
import torchaudio.transforms as TT



import logging
from typing import Optional

logger = logging.getLogger(__name__)

def init_block(fin, fout,kernel_size=(3, 3),padding=(1, 1),stride=(2, 2)):
    return nn.Sequential(
        nn.Conv2d(in_channels=fin, out_channels=fout, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(fout),
        nn.LeakyReLU(),
        )

class model_segment_gru(nn.Module):
    def __init__(self):
        super(model_segment_gru, self).__init__()
        self.conv_block_1 = init_block(1, 32,kernel_size=(2,2),padding=(1,1),stride=(2,2))
        self.conv_block_2 = init_block(32, 32,kernel_size=(2,2),padding=(1,1),stride=(2,2))
        self.conv_block_3 = init_block(32, 64,kernel_size=(2,2),padding=(1,1),stride=(2,2))
        self.conv_block_4 = init_block(64, 64,kernel_size=(2,2),padding=(1,1),stride=(2,2))
        # self.conv_block_5 = init_block(256, 256,kernel_size=(1,9),padding=(0,4),stride=(1,1))
        self.pool1=nn.MaxPool2d(kernel_size=(2,2))
        self.pool2=nn.MaxPool2d(kernel_size=(2,2))
        # self.pool3=nn.MaxPool2d(kernel_size=(2,2))
        self.drop=nn.Dropout(0.1)
        # self.GRU = nn.GRU(256,128,bidirectional=True, batch_first=True)
        # self.fc = nn.Linear(64,32)
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.headMurmur = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 2))
        self.attn = nn.Sequential(nn.Linear(64, 32),   nn.LeakyReLU(), nn.Linear(32, 2),nn.Softmax(dim=1))

    def get_logmel(
            self,
            source: torch.Tensor,
            args=None,
    ) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            # waveform = waveform.unsqueeze(0) * 2 ** 15  # wavform × 2^15
            waveform = waveform.unsqueeze(0)
            # spec = transforms.MelSpectrogram(sr=16000, n_fft=512, win_length=50,
            #                                  hop_length=25, n_mels=128, f_min=25, f_max=2000)(waveform)
            # spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
            fbank = ta_kaldi.fbank(
                waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbank_mean = fbank.mean()
            fbank_std = fbank.std()
            fbank = (fbank - fbank_mean) / (2 * fbank_std)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        return fbank

    def __call__(self, x):
        logmel=self.get_logmel(x)
        x = logmel.unsqueeze(1)
        x = self.conv_block_1(x)
        x=self.pool1(x)
        x = self.conv_block_2(x) 
        # x=self.pool2(x)
        x=self.drop(x)
        x = self.conv_block_3(x)
        x=self.drop(x)
        x = self.conv_block_4(x) 
        # x = x.mean(dim=1)
        x=self.ap(x)
        x = x.squeeze()
        # y = self.fc(x)
        # y = y.transpose(2,0)
        # # z = torch.cat([y,x],dim=1)
        x = self.attn(x)
        # # attn = attn.transpose(2,1)
        # z = self.pool(x*attn).squeeze(2)
        # murmur = self.headMurmur(attn)
        return  x
