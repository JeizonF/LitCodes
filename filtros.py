import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class Filtros:

    def __init__(self, fs=250):

        self.fs = fs
        self.fn = fs / 2

    ##############################################################
    # Remove Offset (DC)
    ##############################################################

    def remover_offset(self, sinal):

        sinal = np.asarray(sinal)

        return sinal - np.mean(sinal)

    ##############################################################
    # High Pass
    ##############################################################

    def highpass(self, sinal, corte=0.5, ordem=2):

        b, a = butter(ordem,
                      corte / self.fn,
                      btype="high")

        return filtfilt(b, a, sinal)

    ##############################################################
    # Low Pass
    ##############################################################

    def lowpass(self, sinal, corte=30, ordem=2):

        b, a = butter(ordem,
                      corte / self.fn,
                      btype="low")

        return filtfilt(b, a, sinal)

    ##############################################################
    # Band Pass
    ##############################################################

    def bandpass(self,
                 sinal,
                 low=0.5,
                 high=15,
                 ordem=2):

        sinal = self.highpass(sinal,
                              corte=low,
                              ordem=ordem)

        sinal = self.lowpass(sinal,
                             corte=high,
                             ordem=ordem)

        return sinal

    ##############################################################
    # Notch 60 Hz
    ##############################################################

    def notch(self,
              sinal,
              freq=60,
              qualidade=30):

        b, a = iirnotch(freq,
                        qualidade,
                        self.fs)

        return filtfilt(b, a, sinal)

    ##############################################################
    # Pipeline completo
    ##############################################################

    def processar(self, sinal):

        sinal = self.remover_offset(sinal)

        sinal = self.notch(sinal)

        sinal = self.bandpass(sinal)

        return sinal