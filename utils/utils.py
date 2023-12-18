import os
from utils.sift import compute_sift_matches
import pandas as pd
import cv2
from natsort import natsorted
from tqdm import tqdm

def compute_matches_in_path_list(files_list, csv_folder='results/'):
    if not isinstance(files_list, list):
        raise TypeError('files_list is not a list')
    
    os.makedirs(csv_folder, exist_ok=True)
    
    csv_name=files_list[0].split('/')[-4]+'_'+files_list[0].split('/')[-3]+'.csv'

    csv_path=os.path.join(csv_folder, csv_name)
    # print("*"*100)
    
    img1_name = files_list[0].split('/')[-1]
    img1 = cv2.imread(files_list[0], cv2.IMREAD_GRAYSCALE)  # queryImage
    
    match_dict={}
        
    for file_counter in tqdm(range(len(files_list))):
        # print(f'file: {files_list[file_counter]}')
        img2_name = files_list[file_counter].split('/')[-1]
        # print(f"File Name {img2_name}")
        
        img2 = cv2.imread(files_list[file_counter], cv2.IMREAD_GRAYSCALE) # trainImage
        matches = compute_sift_matches(img1,img2)
        
        match_dict[img2_name] = matches
        
        
    new_df = pd.DataFrame(match_dict, index=[img1_name])

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col='Unnamed: 0', sep=',').sort_index()
        new_df = pd.concat([df, new_df], axis=0, sort=False)        
        new_df = new_df.reindex(sorted(df.columns), axis=1)

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