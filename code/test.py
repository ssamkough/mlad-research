import numpy as np
from sklearn import preprocessing, neighbors
import pandas as pd
import random
import csv

input_path = "../datasets/cicids17_dataset.csv"

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

n = sum(1 for line in open(input_path))
s = 100000
skip = sorted(random.sample(range(1, n+1), n-s))
df = pd.read_csv(input_path, header = None, names = features, skiprows = skip)
print("Labels:\n", df.pop('label').unique()) # count labels
print("\n---------------------------\n")
