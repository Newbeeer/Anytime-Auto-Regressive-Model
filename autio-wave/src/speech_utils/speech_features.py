 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

import numpy as np
import librosa
from python_speech_features.base import mfcc, logfbank
from python_speech_features import delta
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.io.wavfile import write

class SpeechFeatures(object):

    default_rate = 16000
    default_filters_number = 13
    default_augmented = True

    @staticmethod
    def mfcc(signal):
        return signal

    @staticmethod
    def logfbank(signal, rate=default_rate, filters_number=default_filters_number, augmented=default_augmented):
        logfbank_features = logfbank(signal, rate, nfilt=filters_number)
        if not augmented:
            return logfbank_features
        d_logfbank_features = delta(logfbank_features, 2)
        a_logfbank_features = delta(d_logfbank_features, 2)
        concatenated_features = np.concatenate((
                logfbank_features,
                d_logfbank_features,
                a_logfbank_features
            ),
            axis=1
        )
        return concatenated_features

    @staticmethod
    def features_from_name(name, signal,one_hot, rate=default_rate, filters_number=default_filters_number, augmented=default_augmented):
        return getattr(SpeechFeatures, name)(signal,one_hot,rate, filters_number, augmented)
