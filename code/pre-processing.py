import numpy as np
from sklearn import preprocessing, neighbors
import pandas as pd
import random
import warnings

warnings.filterwarnings('ignore')

# grabbing the sample size
def grab_sample_size (input_path, features):
  yesorno = input("\nWould you like to grab a sample of the data from " + input_path + "?\n")
  if yesorno == "yes":
    n = sum(1 for line in open(input_path)) - 1 # number of records in file (excludes header)
    s = input("\nWhat is your desired sample size?\n") # desired sample size
    s = int(s)
    skip = sorted(random.sample(range(1, n+1), n-s)) # the 0-indexed header will not be included in the skip list
    df = pd.read_csv(input_path, header = None, names = features, skiprows = skip)
  elif yesorno == "1000":
    n = sum(1 for line in open(input_path)) - 1
    s = 1000
    skip = sorted(random.sample(range(1, n+1), n-s))
    df = pd.read_csv(input_path, header = None, names = features, skiprows = skip)
  else:
    df = pd.read_csv(input_path, header = None, names = features)

  return df

# convert categorical features to numerical vectors
# generate 5-class labels
def preprocess_five_class (input_path):
  features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
          'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
          'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
          'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
          'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
          'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
          'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
          'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

  dos_attacks = ['back', 'neptune', 'smurf', 'teardrop', 'land', 'pod', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
  prob_attacks = ['satan', 'portsweep', 'ipsweep', 'nmap', 'mscan', 'saint']
  r2l_attacks = ['warezmaster', 'warezclient', 'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'sendmail', 'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm']
  u2r_attacks = ['rootkit', 'buffer_overflow', 'loadmodule', 'perl', 'httptunnel', 'ps', 'sqlattack', 'xterm']

  df = grab_sample_size(input_path, features)

  for x in dos_attacks:
      df['label'].replace(x+'.', 'dos', inplace=True)
  for x in prob_attacks:
      df['label'].replace(x+'.', 'prob', inplace=True)
  for x in r2l_attacks:
      df['label'].replace(x+'.', 'r2l', inplace=True)
  for x in u2r_attacks:
      df['label'].replace(x+'.', 'u2r', inplace=True)
      
  categorical_columns = ['protocol_type', 'service', 'flag', 'label']
  df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
  for i in categorical_columns:
      df[i] = df[i].astype('int64')

  y = df.pop('label')
  x = df
  
  return x, y

train_file = "../datasets/kddcup.data_10_percent_corrected" # training_small
test_file = "../datasets/corrected" # testing_small

X_train, y_train = preprocess_five_class(train_file) # preprocess_categorical_five_class(train_file)
X_test, y_test =  preprocess_five_class(test_file) # preprocess_categorical_five_class(test_file)
