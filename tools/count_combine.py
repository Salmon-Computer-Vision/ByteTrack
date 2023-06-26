#!/usr/bin/env python3

import logging
import argparse
import os
import os.path as osp

import pandas as pd
import glob

COL_FILENAME = 'Filename'
COL_COUNTABLE_ID = 'Countable ID'
COL_TRACK_ID = 'Track ID'
COL_TOP = 'Top Coord'
COL_LEFT = 'Left Coord'
COL_WIDTH = 'Width'
COL_HEIGHT = 'Height'
COL_CONF = 'Confidence'
COL_FRAME_NUM = 'Frame Num'
COL_DIRECTION = 'Direction'
VAL_LEFT = 'Left'
VAL_RIGHT = 'Right'
COL_COUNT = 'Count'

COL_CLASS_ID = 'Class ID'
COL_XCENTRE = 'x_centre'
COL_YCENTRE = 'y_centre'

logging.basicConfig(level=logging.INFO,
format='%(asctime)s %(levelname)s %(message)s')

def combine_mot_det(opt):
    logging.info("Combining classifications and MOT boxes")
    mot_csvs = glob.glob(osp.join(opt.mot_folder, '**', '*tracks.csv'))
    mot_csvs += glob.glob(osp.join(opt.mot_folder, '*tracks.csv'))
    det_csvs = glob.glob(osp.join(opt.det_folder, '**', '*.txt'))
    det_csvs += glob.glob(osp.join(opt.det_folder, '*.txt'))

    # Create a map with the filename
    mot_csvs_map = [(osp.splitext(osp.basename(csv))[0], csv) for csv in mot_csvs]
    det_csvs_map = [(osp.splitext(osp.basename(csv))[0], csv) for csv in det_csvs]
    mot_csvs_map.sort(key=lambda tup: tup[0])
    det_csvs_map.sort(key=lambda tup: tup[0])

    size = opt.size.split('x')

    det_ind = 0
    for csv_tup in mot_csvs_map: 
        logging.info(csv_tup)

        filename = csv_tup[0]
        filepath = csv_tup[1]

        det_filepath = ''
        while det_ind < len(det_csvs_map):
            if filename.startswith(det_csvs_map[det_ind][0]):
                det_filepath = det_csvs_map[det_ind][1]
                break
            det_ind += 1

        df_mot = pd.read_csv(filepath, index_col=False, names=[COL_FRAME_NUM, COL_TRACK_ID, COL_LEFT, COL_TOP, COL_WIDTH, COL_HEIGHT, COL_CONF])
        df_det = pd.read_csv(det_filepath, sep=' ', index_col=False, names=[COL_CLASS_ID, COL_XCENTRE, COL_YCENTRE, COL_WIDTH, COL_HEIGHT, COL_CONF, COL_FRAME_NUM])

        # Convert YOLO format to MOT sequence format

        # Calculate IOU between the two per frame

        # Assign class ID to each tracking box within the IOU threshold
        return

def count(opt):
    df_counts = pd.DataFrame(columns=[COL_FILENAME, COL_COUNT]).set_index(COL_FILENAME)
    for csv in glob.glob(os.path.join(opt.mot_folder, '**', '*counts.csv'), recursive=True):
        logging.info(csv)
        df = pd.read_csv(csv, names=[COL_FILENAME, COL_FRAME_NUM, COL_COUNTABLE_ID, COL_DIRECTION])

        count = 0
        if not df.empty:
            if opt.upstream != None:
                seen = []
                keep_ids = []
                for _, row in df.sort_values(COL_FRAME_NUM)[::-1].iterrows(): # Traverse in reverse order
                    if row[COL_COUNTABLE_ID] in seen:
                        continue
                    seen.append(row[COL_COUNTABLE_ID])
                    if opt.upstream == 'left' and row[COL_DIRECTION] == VAL_LEFT:
                        keep_ids.append(row[COL_COUNTABLE_ID])
                    if opt.upstream == 'right' and row[COL_DIRECTION] == VAL_RIGHT:
                        keep_ids.append(row[COL_COUNTABLE_ID])
                df = df[df[COL_COUNTABLE_ID].isin(keep_ids)]

            # TODO: Pivot on frame number, too (Detections of same ID and same frame should not happen)
            df_direct = df.pivot_table(index=COL_COUNTABLE_ID, 
                    columns=COL_DIRECTION, values=COL_FRAME_NUM, aggfunc='count')

            if  all(val in df_direct.columns for val in [VAL_LEFT, VAL_RIGHT]):
                count = df_direct[df_direct[VAL_LEFT] == df_direct[VAL_RIGHT]][VAL_LEFT].count()

        df_counts.loc[csv] = count

    df_counts.to_csv(opt.output_csv)
    logging.info(f"Saved counts to {opt.output_csv}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Given countable CSV files from the MOT, generates summarized counts from them.")

  parser.add_argument('-mf', '--mot-folder', default='mot_tracks', help='Path to the folder with the MOT tracks CSV files.')
  parser.add_argument('-o', '--output-csv', default='all_counts.csv', help='Path to the output csv file')
  parser.add_argument('-u', '--upstream', default=None, choices=['right', 'left'], help='Specify upstream direction.')

  subp = parser.add_subparsers(dest='cmd')

  combine_mot_det_p = subp.add_parser('combine', help='Combines class detections and MOT boxes.')
  combine_mot_det_p.add_argument('-df', '--det-folder', default='det_tracks', help='Path to the folder with the class detections .txt files.')
  combine_mot_det_p.add_argument('-s', '--size', default='1920x1080', help='Resolution size.')

  opt = parser.parse_args()
  print(opt, end='\n\n')

  if opt.cmd == 'combine':
      combine_mot_det(opt)
  else:
      count(opt)
