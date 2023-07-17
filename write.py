import json 
import sqlite3
import re
import argparse
import numpy as np

conn = sqlite3.connect('.\\database\\database\\db.sqlite3')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS mosegnas_mosegnasresult
             (
             id INT primary key,
             config TEXT,
             params REAL,
             flops REAL,
             latency REAL,
             FPS REAL,
             mIoU REAL
             )
''')

meta_data = json.load(open("f:/EVO/seg_nas_codes/data/ofa_fanet_plus_bottleneck_rtx_fps@0.5.json", "r"))
# config = np.array([d['config'] for d in meta_data])

# params = np.array([d['params'] for d in meta_data])
# flops = np.array([d['flops'] for d in meta_data])
# latency = np.array([d['latency'] for d in meta_data])
# FPS = np.array([d['FPS'] for d in meta_data])
# mIoU = np.array([d['mIoU'] for d in meta_data])

for index, item in enumerate(meta_data):
    config  = str(item['config'])
    params = float(item['params'])
    flops = float(item['flops'])
    latency = float(item['latency'])
    FPS = float(item['FPS'])
    mIoU = float(item['mIoU'])
    temp = cursor.execute('''
 INSERT OR REPLACE INTO mosegnas_mosegnasresult (
             id,
             config,
             params,
             flops,
             latency,
             FPS,
             mIoU
             ) VALUES
             (?, ?, ?, ?, ?, ?, ?) '''
             , (index, config, params, flops, latency, FPS, mIoU)
                 )
    # pass
conn.commit()
conn.close()
