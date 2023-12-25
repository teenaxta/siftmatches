from utils.utils import create_file_splits, merge_csvs,compute_matches_in_path_list
import concurrent 
from concurrent import futures
import argparse
import multiprocessing
import os


def compute_sift_dataframes(dataset_folder, max_workers):
    splits = create_file_splits(dataset_folder)
    
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit each process with its respective input
        future_results = {executor.submit(compute_matches_in_path_list, data): data for data in splits}
        
def compute_combined_csv(path):
    for csv_folder_paths in os.scandir(path):
        merge_csvs(csv_folder_paths)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Give folder path and number of workers')
    parser.add_argument('--path', default=None, type=str, help='Path to folder with all images')
    parser.add_argument('--max_workers','-w', type=int, default=4, help='Set number of max workers')    
    
    args = parser.parse_args()
    
    folder_path = args.path
    max_workers = args.max_workers    
    
    FOLDERS = [
        '/home/rohan/dataset/Tarten Air/abandonedfactory_night_easy/abandonedfactory_night_easy/P013/image_left/',
        '/home/rohan/dataset/Tarten Air/abandonedfactory_easy/abandonedfactory_easy/P010/image_left/',
        '/home/rohan/dataset/Tarten Air/abandonedfactory_easy/abandonedfactory_easy/P011/image_left/',
    ]
    
    for folder in FOLDERS:
        print(folder)
        compute_sift_dataframes(folder, max_workers)
        
        
    for dataframes_folder in os.scandir('results'):
        compute_combined_csv(dataframes_folder.path)
    
    
