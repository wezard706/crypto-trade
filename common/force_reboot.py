import os
import sys
import logging
from datetime import datetime

# logger
logger = logging.getLogger('force_reboot')
logger.setLevel(logging.DEBUG)
format = logging.Formatter('[%(levelname)s] %(asctime)s, %(message)s')
# 標準出力
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(format)
logger.addHandler(stream_handler)
# ファイル出力(debug)
dfile_handler = logging.FileHandler('common/log/debug.log', 'a')
dfile_handler.setLevel(logging.DEBUG)
dfile_handler.setFormatter(format)
logger.addHandler(dfile_handler)

def force_reboot(log_dir, reboot_file):
    if not os.path.exists(log_dir):
        return

    with open(log_dir) as f:
        debug_log = [line for line in f.readlines() if '[DEBUG]' in line]

    latest_log = debug_log[-1]
    split_log = latest_log.split(',')
    timestamp = split_log[0].replace('[DEBUG] ', '')
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

    now = datetime.now()
    diff = now - timestamp
    if diff.seconds > 300:
        logger.debug(reboot_file + ' process reboot')
        os.execv(sys.executable, [sys.executable] + [reboot_file])
    else:
        logger.debug(reboot_file + ' no problem')

if __name__=='__main__':
    args = sys.argv
    log_dir = args[1]
    reboot_file = args[2]
    force_reboot(log_dir, reboot_file)