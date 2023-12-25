import pathlib
import os
from utils.sift import compute_sift_matches
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
import cv2

def compute_matches_in_path_list(files_list, csv_folder='results/'):
    if not isinstance(files_list, list):
        raise TypeError('files_list is not a list')
    
    csv_folder_name=files_list[0].split('/')[-4]+'_'+files_list[0].split('/')[-3]
    csv_folder_path = f"{csv_folder}/{csv_folder_name}"
    os.makedirs(csv_folder_path, exist_ok=True)

    img1_name = files_list[0].split('/')[-1]
    
    csv_path=os.path.join(csv_folder_path, img1_name.split('.')[0]+'.csv')
        
    img1 = cv2.imread(files_list[0], cv2.IMREAD_GRAYSCALE)  # queryImage
    
    match_dict={}
        
    for file_counter in tqdm(range(len(files_list))):
        img2_name = files_list[file_counter].split('/')[-1]
        
        img2 = cv2.imread(files_list[file_counter], cv2.IMREAD_GRAYSCALE) # trainImage
        matches = compute_sift_matches(img1,img2)
        
        match_dict[img2_name] = matches
        
    new_df = pd.DataFrame(match_dict, index=[img1_name])
    new_df.to_csv(csv_path, index=True)
    
    print("*"*100)
    print('Complete')
    
def create_file_splits(PATH):
    files = natsorted(os.listdir(PATH))
    splits=[]
    for i in range (len(files)):
        splits.append(files[i:])
        
    splits = [[os.path.join(PATH, file_name) for file_name in split] for split in splits]
    
    return splits

def merge_csvs(folder_path):
    combined_folder_path = os.path.join('final_results','combined/')
    os.makedirs(combined_folder_path, exist_ok=True)
    
    file_name = folder_path.split('/')[-2]+'.csv'
    csv_path  = os.path.join(combined_folder_path, file_name)
    file_paths = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file_path) for file_path in file_paths]
    
    df = pd.DataFrame()
    for file_path in tqdm(file_paths):
        new_df = pd.read_csv(file_path, index_col='Unnamed: 0', sep=',').sort_index()
        df = pd.concat([df, new_df])
    
    df.to_csv(csv_path, index=True)        
    
    
    
    
    
    
    
    