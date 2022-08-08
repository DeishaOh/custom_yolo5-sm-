#jupyter notebook을 못보는 경우 사용


!git clone https://github.com/ultralytics/yolov5
 
cd yolov5-master

!pip install -r requirements.txt

# 본인 데이터 확인
from glob import glob
img_list = glob('C:/Users/rk401/arglass/dataset/images/*.jpg')
print(len(img_list))

# train - val을 나누었다면 split을 할 필요가 없음
from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)
print(len(train_img_list), len(val_img_list))

with open('C:/Users/rk401/arglass/dataset/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('C:/Users/rk401/arglass/dataset/val.txt', 'w') as f:
    f.write('\n'.join(val_img_list) + '\n')
    
    
# .yaml 파일 내의 경로 고치기
# vscode. txt 등으로 고쳐도 됨
import yaml
with open('C:/Users/rk401/arglass/dataset/data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
print(data)

data['train'] = 'C:/Users/rk401/arglass/dataset/train.txt'
data['val'] = 'C:/Users/rk401/arglass/dataset/val.txt'

with open('C:/Users/rk401/arglass/dataset/data.yaml', 'w') as f:
    yaml.dump(data, f)
    

#학습 시작
%cd C:/Users/rk401/arglass/yolov5-master/
!python train.py --img 416 --batch 8 --epochs 50 --data C:/Users/rk401/arglass/dataset/data.yaml --cfg C:/Users/rk401/arglass/yolov5-master/models/yolov5s.yaml --weights yolov5s.pt --name results
print(data)


#결과 확인
#source에 확인할 파일 경로 지정
#0은 
from IPython.display import Image
import os
!python detect.py  --weights C:/Users/rk401/arglass/yolov5-master/runs/train/results3/weights/best.pt --img 416 --conf 0.3 --source 0
#Image(os.path.join('/content/yolov5/inference/output', os.path.basename(test_img_path)))
