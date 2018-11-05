import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_file = "../../datasets/kddcup.data_10_percent_corrected" # training_small
test_file = "../../datasets/corrected" # testing_small

def preprocess_five_class (input_path):
  features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
          'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
          'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
          'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
          'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
          'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
          'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
          'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

  df = pd.read_csv(input_path, header = None, names = features)

  dos_attacks = ['back', 'neptune', 'smurf', 'teardrop', 'land', 'pod', 'apache2', 'mailbomb', 'processtable', 'udpstorm']
  prob_attacks = ['satan', 'portsweep', 'ipsweep', 'nmap', 'mscan', 'saint']
  r2l_attacks = ['warezmaster', 'warezclient', 'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'sendmail', 'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm']
  u2r_attacks = ['rootkit', 'buffer_overflow', 'loadmodule', 'perl', 'httptunnel', 'ps', 'sqlattack', 'xterm']

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

  df_x = data.iloc[:,1:]
  df_y = data.iloc[:,0]
  
  y = df.pop('label')
  x = df
  
  return x, y

x_train, y_train = preprocess_five_class(train_file) # preprocess_categorical_five_class(train_file)
x_test, y_test =  preprocess_five_class(test_file) # preprocess_categorical_five_class(test_file)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
pred

s = y_test.values
count = 0

for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1

len(pred)
