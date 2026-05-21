import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import glob
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class myDataset(Dataset):
    def __init__(self, dataset_dir, seq_length=30, scaler_dir=None):
        dataset_path = Path(dataset_dir)
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path

        scaler_path = Path(scaler_dir) if scaler_dir else PROJECT_ROOT / "scalers"
        if not scaler_path.is_absolute():
            scaler_path = PROJECT_ROOT / scaler_path

        self.dir_list = sorted(path.name for path in dataset_path.iterdir() if path.is_dir())
        self.dir_name = str(dataset_path)
        self.seq_length = seq_length
        self.scaler_dir = scaler_path
        
        # 스케일러 초기화
        self.temp_std_scaler = StandardScaler()
        self.co_std_scaler = StandardScaler()
        self.soot_std_scaler = StandardScaler()
        self.devc_std_scaler = StandardScaler()
        
        print(f"Found {len(self.dir_list)} directories in {dataset_dir}")
        
        # 스케일러 로드
        # self.standard_scale()  # 스케일러 학습을 위해 주석 해제
        self.temp_std_scaler = joblib.load(self.scaler_dir / 'temp_scaler.pkl')
        self.co_std_scaler = joblib.load(self.scaler_dir / 'co_scaler.pkl')
        self.soot_std_scaler = joblib.load(self.scaler_dir / 'soot_scaler.pkl')
        self.devc_std_scaler = joblib.load(self.scaler_dir / 'devc_scaler.pkl')
        
        # 유효한 (폴더, 인덱스) 쌍 저장
        self.valid_samples = []
        
        for folder_idx, dir_name in enumerate(self.dir_list):
            folder_path = os.path.join(self.dir_name, dir_name)
            
            # 슬라이스 파일 목록 가져오기
            slice_files = glob.glob(os.path.join(folder_path, "slice_*.csv"))
            
            # 센서 데이터 파일이 존재하는지 확인
            devc_file = os.path.join(folder_path, f"{dir_name}_devc.csv")
            if not os.path.exists(devc_file):
                print(f"Warning: No sensor data found for {dir_name}")
                continue
                
            # 센서 데이터 파일의 행 수 확인
            try:
                devc_data = pd.read_csv(devc_file, header=1)
                if len(devc_data) < self.seq_length:
                    print(f"Warning: Not enough sensor data in {dir_name}")
                    continue
            except Exception as e:
                print(f"Error reading devc file {devc_file}: {e}")
                continue
            
            # 슬라이스 파일 인덱스 추출
            slice_indices = []
            for file_path in slice_files:
                try:
                    file_name = os.path.basename(file_path)
                    # slice_60_61.csv 형식의 파일명에서 인덱스 추출
                    parts = file_name.split('_')
                    if len(parts) >= 3 and parts[0] == "slice" and parts[1].isdigit():
                        slice_indices.append(int(parts[1]))
                except Exception as e:
                    print(f"Error parsing slice file name {file_path}: {e}")
            
            # 인덱스 정렬
            slice_indices = sorted(slice_indices)
            
            if not slice_indices:
                print(f"Warning: No valid slice files found in {dir_name}")
                continue
                
            # 유효한 인덱스 식별
            max_devc_idx = len(devc_data) - self.seq_length
            
            # 각 가능한 시작 인덱스에 대해 확인
            for idx in range(min(90, max_devc_idx + 1)):  # 원래 로직에서는 0~89 범위 가정
                label_idx = idx + 60  # 예측하려는 시점은 60 스텝 이후
                
                # 해당 인덱스의 슬라이스 파일이 존재하는지 확인
                if label_idx in slice_indices:
                    self.valid_samples.append((folder_idx, idx))
                    
        print(f"Found {len(self.valid_samples)} valid samples across all directories")
    
    def __len__(self):
        return len(self.valid_samples)
        
    def __getitem__(self, idx):
        if idx >= len(self.valid_samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.valid_samples)} samples")
            
        folder_idx, sample_idx = self.valid_samples[idx]
        dir_name = self.dir_list[folder_idx]
        
        # 센서 데이터 로드 및 전처리
        try:
            data = pd.read_csv(os.path.join(f"{self.dir_name}/{dir_name}/", 
                                          f"{dir_name}_devc.csv"), header=1)
            data = data.values.astype(np.float32)
            data = data[:, 1:71]  # 첫 번째 열(시간) 제외하고 70개 센서 데이터 사용
            
            # 데이터 재구성
            re_data = []
            for row in data:
                row_2d = row.reshape(7, 10)
                row_2d[1:] = np.fliplr(row_2d[1:])
                row_2d[1:, [-2, -1]] = row_2d[1:, [-1, -2]]
                row_2d[1:, [-2, -3]] = row_2d[1:, [-3, -2]]
                transformed_row = row_2d.reshape(-1)
                re_data.append(transformed_row)
            
            data = np.array(re_data) 
            data = data[sample_idx:sample_idx+self.seq_length]
            # 센서 데이터 스케일링
            data = self.devc_std_scaler.transform(data)
        except Exception as e:
            print(f"Error processing sensor data for {dir_name} at index {sample_idx}: {e}")
            # 재귀적으로 다른 샘플 시도 (깊이 제한을 위해 최대 5회)
            return self.__getitem__((idx + 1) % len(self))
        
        # 슬라이스 데이터 로드
        label_idx = sample_idx + 60
        try:
            label_data = pd.read_csv(os.path.join(f"{self.dir_name}/{dir_name}/", 
                                               f"slice_{label_idx}_{label_idx + 1}.csv"), 
                                  sep=", ", 
                                  names=['X', 'Y', 'TEMPERATURE', 'SOOT_VISIBILITY', 'CO_FRACTION'],  
                                  engine="python",
                                  skiprows=2)
            
            # 각 변수별 개별 스케일링
            temp_values = label_data["TEMPERATURE"].values.astype(np.float32).reshape(-1, 1)
            co_values = label_data["CO_FRACTION"].values.astype(np.float32).reshape(-1, 1)
            soot_values = label_data["SOOT_VISIBILITY"].values.astype(np.float32).reshape(-1, 1)
            
            temp_scaled = self.temp_std_scaler.transform(temp_values).reshape(53, 66)
            co_scaled = self.co_std_scaler.transform(co_values).reshape(53, 66)
            soot_scaled = self.soot_std_scaler.transform(soot_values).reshape(53, 66)
            
            # 각 스케일링된 데이터 flip
            temp_scaled = np.flip(temp_scaled, axis=0)
            co_scaled = np.flip(co_scaled, axis=0)
            soot_scaled = np.flip(soot_scaled, axis=0)
            
            # 최종 데이터 형태로 재구성
            total_data = np.stack([temp_scaled, co_scaled, soot_scaled])
        except Exception as e:
            print(f"Error processing slice data for {dir_name} at label index {label_idx}: {e}")
            # 재귀적으로 다른 샘플 시도
            return self.__getitem__((idx + 1) % len(self))
        
        # 텐서 변환
        data = torch.tensor(data, dtype=torch.float32)
        total_data = torch.tensor(total_data, dtype=torch.float32)
        
        return data, total_data

    def standard_scale(self):
        # 각 변수별 데이터 수집
        all_temp_data = []
        all_co_data = []
        all_soot_data = []
        all_devc_data = []
        
        # 각 폴더에 대해 반복
        for dir_name in self.dir_list:
            folder_path = os.path.join(self.dir_name, dir_name)
            
            # 센서 데이터 수집
            try:
                devc_file = os.path.join(folder_path, f"{dir_name}_devc.csv")
                devc_data = pd.read_csv(devc_file, header=1)
                devc_values = devc_data.iloc[:, 1:71].values.astype(np.float32)
                
                # 데이터 재구성
                re_devc_data = []
                for row in devc_values:
                    row_2d = row.reshape(7, 10)
                    row_2d[1:] = np.fliplr(row_2d[1:])
                    row_2d[1:, [-2, -1]] = row_2d[1:, [-1, -2]]
                    row_2d[1:, [-2, -3]] = row_2d[1:, [-3, -2]]
                    transformed_row = row_2d.reshape(-1)
                    re_devc_data.append(transformed_row)
                
                all_devc_data.extend(re_devc_data)
            except Exception as e:
                print(f"Error reading devc file {devc_file}: {e}")
            
            # 슬라이스 파일 수집
            slice_files = glob.glob(os.path.join(folder_path, "slice_*.csv"))
            for file_path in slice_files:
                try:
                    slice_data = pd.read_csv(file_path, sep=", ",
                                        names=['X', 'Y', 'TEMPERATURE', 'SOOT_VISIBILITY', 'CO_FRACTION'],
                                        engine="python", skiprows=2)
                    
                    # 각 변수별 데이터 추출 및 수집
                    temp_values = slice_data['TEMPERATURE'].values.astype(np.float32).reshape(-1, 1)
                    co_values = slice_data['CO_FRACTION'].values.astype(np.float32).reshape(-1, 1)
                    soot_values = slice_data['SOOT_VISIBILITY'].values.astype(np.float32).reshape(-1, 1)
                    
                    all_temp_data.append(temp_values)
                    all_co_data.append(co_values)
                    all_soot_data.append(soot_values)
                    
                except Exception as e:
                    print(f"Error reading slice file {file_path}: {e}")
        
        # 각 데이터 병합
        combined_temp_data = np.vstack(all_temp_data) if all_temp_data else np.array([]).reshape(0, 1)
        combined_co_data = np.vstack(all_co_data) if all_co_data else np.array([]).reshape(0, 1)
        combined_soot_data = np.vstack(all_soot_data) if all_soot_data else np.array([]).reshape(0, 1)
        combined_devc_data = np.vstack(all_devc_data) if all_devc_data else np.array([]).reshape(0, 70)
        
        # 각 스케일러 학습
        print(f"Fitting temp scaler with {len(combined_temp_data)} samples")
        self.temp_std_scaler.fit(combined_temp_data)
        
        print(f"Fitting CO scaler with {len(combined_co_data)} samples")
        self.co_std_scaler.fit(combined_co_data)
        
        print(f"Fitting soot scaler with {len(combined_soot_data)} samples")
        self.soot_std_scaler.fit(combined_soot_data)
        
        print(f"Fitting devc scaler with {len(combined_devc_data)} samples")
        self.devc_std_scaler.fit(combined_devc_data)
        
        print("All scalers fitted successfully!")

        os.makedirs(self.scaler_dir, exist_ok=True)
        joblib.dump(self.temp_std_scaler, self.scaler_dir / 'temp_scaler.pkl')
        joblib.dump(self.co_std_scaler, self.scaler_dir / 'co_scaler.pkl')
        joblib.dump(self.soot_std_scaler, self.scaler_dir / 'soot_scaler.pkl')
        joblib.dump(self.devc_std_scaler, self.scaler_dir / "devc_scaler.pkl")
        
        

# def heatmap(data_matrix,title):
#     plt.figure(figsize=(12, 8))
#     heatmap = sns.heatmap(data_matrix[2],cmap="jet")
#     plt.title(title)
#     # plt.gca().invert_yaxis()
#     plt.savefig('fig1.png')



def visualize_images(true_image):
    # PyTorch 텐서를 NumPy 배열로 변환하여 시각화
    true_image_np = true_image.permute(2,0,1).cpu().numpy()  # (C, H, W) → (H, W, C)
    
    plt.imshow(true_image_np)  # 이미지를 출력
    plt.title("True Image")    # 그래프 제목 설정
    plt.axis("on")            # 축 숨김
    plt.show() 

if __name__ == "__main__":
    A=myDataset("data/train_data")
    input_data,total_data = A[300]
    
    # heatmap(total_data,"kimmin")
    
        
