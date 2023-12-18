from utils.utils import create_file_splits, compute_matches_in_path_list
import concurrent 
from concurrent import futures
import argparse


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Give folder path and number of workers')
    parser.add_argument('path', type=str, help='Path to folder with all images')
    parser.add_argument('--max_workers','-w', type=int, default=4, help='Set number of max workers')    
    
    args = parser.parse_args()
    
    folder_path = args.path
    max_workers = args.max_workers    
    
    # PATH = "/home/rohan/dataset/Tarten Air/abandonedfactory_easy/abandonedfactory_easy/P008/image_left/"
    splits = create_file_splits(folder_path)
    # splits=splits[0]
    
    # with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(compute_matches_in_path_list, splits)
    
    
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit each process with its respective input
        future_results = {executor.submit(compute_matches_in_path_list, data): data for data in splits}

        # Iterate through the futures as they complete
        for future in futures.as_completed(future_results):
            input_data = future_results[future]
            try:
                # Get the result of the completed future
                result = future.result()
                # print(f"Input: {input_data}, Result: {result}")
            except Exception as e:
                # print(f"Input: {input_data}, Exception: {e}")
                pass