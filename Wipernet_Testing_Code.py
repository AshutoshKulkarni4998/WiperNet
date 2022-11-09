from __future__ import absolute_import, division, print_function, unicode_literals

try:
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import time
import sys
from absl import app
from tqdm import tqdm
from matplotlib import pyplot as plt
from IPython import display
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,Conv2DTranspose, AveragePooling2D, Lambda, Add, ReLU, MaxPooling2D,Conv2D,Add,Subtract,Multiply,PReLU,GlobalAveragePooling2D
import datetime
import numpy as np
from termcolor import colored, cprint

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
nf=8
LAMBDA = 10

################################# DATASET PATHS ###########################################


Mode = 'test'
validation_data ='./Test_Data'
test_folder_name = 'Results'
checkpoint_load = 'Yes'
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  input_image = tf.cast(image, tf.float32)
  

  return input_image

def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                method=tf.image.ResizeMethod.BILINEAR)
  
  return input_image

def normalize(input_image):
  input_image = (input_image / 127.5) - 1
  

  return input_image

@tf.function()
def random_jitter(input_image):
  
  input_image = resize(input_image,256, 256)

  
  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    

  return input_image

def load_image_test(image_file):
  input_image= load(image_file)
  
  # input_image = resize(input_image,IMG_HEIGHT, IMG_WIDTH)
  input_image = normalize(input_image)

  return input_image




generator1=tf.saved_model.load("WiperNet_Model")

class GAN(object):
  
  def __init__(self, mode,output_path):
  
    self.output_path = output_path
    os.path.join(self.output_path)
    self.gen1 = generator1
    self.generator1 = generator1
    

    self.checkpoint_dir1 = self.output_path + './training_checkpoints/' + 'gen1'
    self.checkpoint_prefix1 = os.path.join(self.checkpoint_dir1, "ckpt")
    self.checkpoint1 = tf.train.Checkpoint(
                     generator1=self.generator1
                     )

  

  def generate_images(self, test_input, number, mode=Mode):
    
    if mode == 'test' :
      mode = test_folder_name
      derained= self.generator1(test_input, training=True)
      
      
      display_list = [derained[0]]
      
      image = np.hstack([img for img in display_list])
      try :
        os.mkdir(self.output_path+'/{}'.format(mode))
      except:
        pass
      plt.imsave(self.output_path+'/{}/{}_.jpg'.format(mode,number), np.array((image * 0.5 + 0.5)*255, dtype='uint8'))
    else:
      print('Enter valid mode eighter [!]train or [!]test')
      exit(0)

    
  def test(self, dataset):
    self.checkpoint1.restore(tf.train.latest_checkpoint(self.checkpoint_dir1)) 
   
    
    text = colored('Checkpoint restored !!!','magenta')
    print(text)
    print(colored('='*50,'magenta'))
    for n, (example_input) in tqdm(dataset.enumerate()):
      self.generate_images(example_input, n, mode='test')
    print(colored("Model Tested Successfully !!!!! ",'green',attrs=['reverse','blink'])) 
  def load_checkpoint(self):
    self.checkpoint1.restore(tf.train.latest_checkpoint(self.checkpoint_dir1))


def run_main(argv):
  del argv
  kwargs = {
      'mode':Mode, 
      'output_path':'WiperNet/',
      'batch_size':1}
  main(**kwargs)

def main(mode,output_path,batch_size):

  gan = GAN(mode,output_path)
  if checkpoint_load == 'Yes':
    gan.load_checkpoint()
    print("############################## Checkpoint Loaded ############################")
  if mode=='test':
    test_dataset =tf.data.Dataset.list_files(validation_data + '/*',shuffle=False)
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(batch_size)
    
    gan.test(test_dataset)
  print("############################## TESTING COMPLETED ############################")

if __name__ == '__main__':
  app.run(run_main)