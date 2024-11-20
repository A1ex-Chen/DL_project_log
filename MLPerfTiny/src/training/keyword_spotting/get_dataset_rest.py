#!/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.platform import gfile

import functools

import matplotlib.pyplot as plt
import numpy as np
import os, pickle

import kws_util
import keras_model as models

word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right",
               "Stop", "Up", "Yes", "Silence", "Unknown"]









    
      # Warp the linear-scale, magnitude spectrograms into the mel-scale.
      lower_edge_hertz, upper_edge_hertz = 0.0, model_settings['sample_rate'] / 2.0
      linear_to_mel_weight_matrix = (
          tf.signal.linear_to_mel_weight_matrix(
              num_mel_bins=num_mel_bins,
              num_spectrogram_bins=num_spectrogram_bins,
              sample_rate=model_settings['sample_rate'],
              lower_edge_hertz=lower_edge_hertz,
              upper_edge_hertz=upper_edge_hertz))

      mel_spectrograms = tf.tensordot(powspec, linear_to_mel_weight_matrix,1)
      mel_spectrograms.set_shape(magspec.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))

      log_mel_spec = 10 * log10(mel_spectrograms)
      log_mel_spec = tf.expand_dims(log_mel_spec, -1, name="mel_spec")
    
      log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
      log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)

      next_element['audio'] = log_mel_spec

    elif model_settings['feature_type'] == 'td_samples':
      ## sliced_foreground should have the right data.  Make sure it's the right format (int16)
      # and just return it.
      paddings = [[0, 16000-tf.shape(sliced_foreground)[0]]]
      wav_padded = tf.pad(sliced_foreground, paddings)
      wav_padded = tf.expand_dims(wav_padded, -1)
      wav_padded = tf.expand_dims(wav_padded, -1)
      next_element['audio'] = wav_padded
      
    return next_element
  
  return prepare_processing_graph


def prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME):
  """Searches a folder for background noise audio, and loads it into memory.
  It's expected that the background audio samples will be in a subdirectory
  named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
  the sample rate of the training data, but can be much longer in duration.
  If the '_background_noise_' folder doesn't exist at all, this isn't an
  error, it's just taken to mean that no background noise augmentation should
  be used. If the folder does exist, but it's empty, that's treated as an
  error.
  Returns:
    List of raw PCM-encoded audio samples of background noise.
  Raises:
    Exception: If files aren't found in the folder.
  """
  background_data = []
  background_dir = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME)
  if not os.path.exists(background_dir):
    return background_data
  #with tf.Session(graph=tf.Graph()) as sess:
  #    wav_filename_placeholder = tf.placeholder(tf.string, [])
  #    wav_loader = io_ops.read_file(wav_filename_placeholder)
  #    wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
  search_path = os.path.join(bg_path, BACKGROUND_NOISE_DIR_NAME,'*.wav')
  #for wav_path in gfile.Glob(search_path):
  #    wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
  #    self.background_data.append(wav_data)
  for wav_path in gfile.Glob(search_path):
    #audio = tfio.audio.AudioIOTensor(wav_path)
    raw_audio = tf.io.read_file(wav_path)
    audio = tf.audio.decode_wav(raw_audio)
    background_data.append(audio[0])
  if not background_data:
    raise Exception('No background wav files were found in ' + search_path)
  return background_data


def get_training_data(Flags, get_waves=False, val_cal_subset=False):
  
  label_count=12
  background_frequency = Flags.background_frequency
  background_volume_range_= Flags.background_volume
  model_settings = models.prepare_model_settings(label_count, Flags)

  bg_path=Flags.bg_path
  BACKGROUND_NOISE_DIR_NAME='_background_noise_' 
  background_data = prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME)

  splits = ['train', 'test', 'validation']
  (ds_train, ds_test, ds_val), ds_info = tfds.load('speech_commands', split=splits, 
                                                data_dir=Flags.data_dir, with_info=True)

  if val_cal_subset:  # only return the subset of val set used for quantization calibration
    with open("quant_cal_idxs.txt") as fpi:
      cal_indices = [int(line) for line in fpi]
    cal_indices.sort()
    # cal_indices are the positions of specific inputs that are selected to calibrate the quantization
    count = 0  # count will be the index into the validation set.
    val_sub_audio = []
    val_sub_labels = []
    for d in ds_val:
      if count in cal_indices:          # this is one of the calibration inpus
        new_audio = d['audio'].numpy()  # so add it to a stack of tensors 
        if len(new_audio) < 16000:      # from_tensor_slices doesn't work for ragged tensors, so pad to 16k
          new_audio = np.pad(new_audio, (0, 16000-len(new_audio)), 'constant')
        val_sub_audio.append(new_audio)
        val_sub_labels.append(d['label'].numpy())
      count += 1
    # and create a new dataset for just the calibration inputs.
    ds_val = tf.data.Dataset.from_tensor_slices({"audio": val_sub_audio,
                                                 "label": val_sub_labels})

  if Flags.num_train_samples != -1:
    ds_train = ds_train.take(Flags.num_train_samples)
  if Flags.num_val_samples != -1:
    ds_val = ds_val.take(Flags.num_val_samples)
  if Flags.num_test_samples != -1:
    ds_test = ds_test.take(Flags.num_test_samples)
    
  if get_waves:
    ds_train = ds_train.map(cast_and_pad)
    ds_test  =  ds_test.map(cast_and_pad)
    ds_val   =   ds_val.map(cast_and_pad)
  else:
    # extract spectral features and add background noise
    ds_train = ds_train.map(get_preprocess_audio_func(model_settings,is_training=True,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test  =  ds_test.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val   =   ds_val.map(get_preprocess_audio_func(model_settings,is_training=False,
                                                      background_data=background_data),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # change output from a dictionary to a feature,label tuple
    ds_train = ds_train.map(convert_dataset)
    ds_test = ds_test.map(convert_dataset)
    ds_val = ds_val.map(convert_dataset)

  # Now that we've acquired the preprocessed data, either by processing or loading,
  ds_train = ds_train.batch(Flags.batch_size)
  ds_test = ds_test.batch(Flags.batch_size)  
  ds_val = ds_val.batch(Flags.batch_size)
  
  return ds_train, ds_test, ds_val


if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()
  ds_train, ds_test, ds_val = get_training_data(Flags)

  for dat in ds_train.take(1):
    print("One element from the training set has shape:")
    print(f"Input tensor shape: {dat[0].shape}")
    print(f"Label shape: {dat[1].shape}")