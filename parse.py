# coding=utf8

import argparse
import torch

parser = argparse.ArgumentParser(description='AMD')

parser.add_argument( '--county-file', type=str, default='./data/county_data.csv', help='county file')
parser.add_argument( '--zcta-file', type=str, default='./data/zcta_data.csv', help='zcta file')
parser.add_argument( '--sdi-zcta-file', type=str, default='./data/zcta_sdi.csv', help='sdi zcta file')
parser.add_argument( '--sdi-county-file', type=str, default='./data/county_sdi.csv', help='sdi county file')
parser.add_argument( '--prediction_window', type=int, default=3, help='prediction window')
parser.add_argument( '--observation_window', type=int, default=10, help='observation window')
parser.add_argument( '--train_tm', type=int, default=80)
parser.add_argument( '--valid_tm', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')



args = parser.parse_args()

args.feature_cols = ['cocaine', 'fentanyl', 'heroin.adj', 'methamphetamine',  'fent.cocaine', 'fent.stim', 'fent.heroin', 'benzo', 'prescribe.opioid',  'alcohol', 'label']
args.sdoh_cols = ['SDI_score',
           'PovertyLT100_FPL_score', 'Single_Parent_Fam_score',
           'Education_LT12years_score', 'HHNo_Vehicle_score',
           'HHRenter_Occupied_score', 'HHCrowding_score', 'Nonemployed_score',
           'sdi']
args.label_col = 'label_per'
args.pop_col = 'population'