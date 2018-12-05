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
# generate 2-class or 5-class labels
def preprocess_five_class (input_path):
  # 85 features
  features = ['flow_id', 'source_ip', 'source_port', 'dest_ip', 'dest_port',
  'protocol', 'timestamp', 'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
  'total_length_fwd_packets', 'total_length_bwd_packets', 'fwd_packet_length_max',
  'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std', 'bwd_packet_length_max',
  'bwd_packet_length_min', 'bwd_packet_length_mean', 'bwd_packet_length_std',
  'flow_bytes', 'flow_packets', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
  'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
  'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
  'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
  'fwd_header_length', 'bwd_header_length', 'fwd_packets', 'bwd_packets', 'min_packet_length',
  'max_packet_length', 'packet_length_mean', 'packet_length_std', 'packet_length_variance',
  'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count',
  'urg_flag_count', 'cwe_flag_count', 'ece_flag_count', 'down_up_ratio', 'avg_packet_size',
  'avg_fwd_segment_size', 'avg_bwd_segment_size', 'fwd_header_length', 'fwd_avg_bytes_bulk',
  'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate',  'bwd_avg_bytes_bulk', 'bwd_avg_packets_bulk',
  'bwd_avg_bulk_rate', 'subflow_fwd_packets', 'subflow_fwd_bytes', 'subflow_bwd_packets',
  'subflow_bwd_bytes', 'init_win_bytes_fwd', 'init_win_bytes_bwd', 'act_data_pkt_fwd',
  'min_seg_size_forward', 'act_mean', 'act_std', 'act_max', 'act_min', 'idle_mean', 'idle_std',
  'idle_max', 'idle_min', 'label']

  # ??? attacks
  attacks = ['back', 'neptune', 'smurf', 'teardrop', 'land', 'pod', 'apache2', 'mailbomb', 'processtable', 'udpstorm',
             'satan', 'portsweep', 'ipsweep', 'nmap', 'mscan', 'saint', 'warezmaster', 'warezclient', 'ftp_write',
             'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'sendmail', 'named', 'snmpgetattack', 'snmpguess', 'xlock',
             'xsnoop', 'worm', 'rootkit', 'buffer_overflow', 'loadmodule', 'perl', 'httptunnel', 'ps', 'sqlattack', 'xterm']
  
  dos_attacks = ['back', 'neptune', 'smurf', 'teardrop', 'land', 'pod', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
  prob_attacks = ['satan', 'portsweep', 'ipsweep', 'nmap', 'mscan', 'saint']
  r2l_attacks = ['warezmaster', 'warezclient', 'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'sendmail', 'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm']
  u2r_attacks = ['rootkit', 'buffer_overflow', 'loadmodule', 'perl', 'httptunnel', 'ps', 'sqlattack', 'xterm']

  df = grab_sample_size(input_path, features)

  running = True
  while (running):
    yesorno = input("\nWould you like to do 2-classification or 5-classification?\n")
    if yesorno == "2":
      for x in attacks:
          df['label'].replace(x+'.', 'attack', inplace=True)
      running = False
    elif yesorno == "5":      
      for x in dos_attacks:
          df['label'].replace(x+'.', 'dos', inplace=True)
      for x in prob_attacks:
          df['label'].replace(x+'.', 'prob', inplace=True)
      for x in r2l_attacks:
          df['label'].replace(x+'.', 'r2l', inplace=True)
      for x in u2r_attacks:
          df['label'].replace(x+'.', 'u2r', inplace=True)
      running = False
    else:
      print("Please either input '2' for 2-classification or '5' for 5-classification.")

  # changing categorical values to numerical values
  categorical_columns = ['protocol_type', 'service', 'flag', 'label']
  df[categorical_columns] = df[categorical_columns].astype('category').apply(lambda x: x.cat.codes)
  for i in categorical_columns:
      df[i] = df[i].astype('int64')

  y = df.pop('label')
  x = df
  
  return x, y

train_file = "../../../datasets/cicids17_dataset.csv"
test_file = "../../../datasets/cicids17_dataset.csv"

x_train, y_train = preprocess_five_class(train_file)
x_test, y_test =  preprocess_five_class(test_file)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test) # takes the features of a record and predicts the label of a record

print("\nMetrics")
print("---------")

accuracy = clf.score(x_test, y_test) * 100
print("Accuracy: " + str(accuracy))

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, average=None)
#for i in recall:
#  recall[i] * 100
print("Recall: " + str(recall))

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, average=None)
#for i in precision:
#  precision[i] * 100
print("Precision: " + str(precision))
