import json 
import sqlite3
import re
import argparse

conn = sqlite3.connect('.\\database\\database\\db.sqlite3')  
conn.execute('''
CREATE TABLE IF NOT EXISTS mosegnas_result
''')