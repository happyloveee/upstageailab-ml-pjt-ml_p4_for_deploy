import warnings
import torch
import torchvision

# 경고 메시지 비활성화
warnings.filterwarnings('ignore')

# torchvision.disable_beta_transforms_warning() 제거

# torchvision image 관련 경고 비활성화
warnings.filterwarnings('ignore', message='Failed to load image Python extension')