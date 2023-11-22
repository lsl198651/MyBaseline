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
        nn.Dropout(p=0.1))

class model_segment_gru(nn.Module):
    def __init__(self):
        super(model_segment_gru, self).__init__()
        self.conv_block_1 = init_block(1, 256,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.conv_block_2 = init_block(256, 256,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.conv_block_3 = init_block(256, 256,kernel_size=(4,4),padding=(1,1),stride=(2,2))
        self.conv_block_4 = init_block(256, 256,kernel_size=(4,4),padding=(1,1),stride=(2,2))
        self.conv_block_5 = init_block(256, 256,kernel_size=(1,9),padding=(0,4),stride=(1,1))
        self.GRU = nn.GRU(256,128,bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256,5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.headMurmur = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU(), nn.Linear(16, 2))
        self.attn = nn.Sequential(nn.Linear(256, 128), nn.LeakyReLU(), nn.Linear(128, 32), nn.Softmax(dim=1))

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
        x=self.get_logmel(x)
        x = x.unsqueeze(0)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x) 
        x = self.conv_block_3(x)
        x = self.conv_block_4(x) 
        #x = F.dropout(x,0.5,self.training)
        x = x.squeeze(0)
        y = x.transpose(2,0)
        y,_ = self.GRU(y)
        # y = self.fc(y)
        # y = y.transpose(2,0)

        # # z = torch.cat([y,x],dim=1)
        attn = self.attn(y)
        # # attn = attn.transpose(2,1)
        # z = self.pool(x*attn).squeeze(2)
        murmur = self.headMurmur(attn)
        return  murmur
