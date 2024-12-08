import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from types import SimpleNamespace

@dataclass
class MLflowConfig:
    """MLflow 관련 설정"""
    tracking_uri: str
    experiment_name: str
    model_registry_metric_threshold: float
    mlrun_path: Path
    backend_store_uri: Path
    model_info_path: Path
    artifact_location: Path
    server_config: Dict[str, Any]

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

class MLflowConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Config:
    def __init__(self, config_path: str = None):
        try:
            load_dotenv()
            
            # Render 환경 확인
            self.is_render = os.getenv('RENDER', 'false').lower() == 'true'
            
            # 기본 설정 먼저 로드
            self._config = self._get_default_config()
            
            # 프로젝트 루트 설정
            self.project_root = self._find_project_root()
            
            # config 파일이 있으면 로드
            if config_path:
                config_file = self.project_root / config_path
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                        self._config.update(file_config)
            
            # 설정 초기화
            self._setup_paths()
            self._setup_mlflow()
            self._setup_project_config()
            self._create_directories()
            
        except Exception as e:
            print(f"Config 초기화 오류: {e}")
            raise

    def _find_project_root(self) -> Path:
        """프로젝트 루트 디렉토리 찾기"""
        if self.is_render:
            return Path(os.getenv('RENDER_PROJECT_ROOT', '/opt/render/project/src'))
        return Path(__file__).parent.parent

    def _get_default_config(self) -> dict:
        """기본 설정값 반환"""
        return {
            "mlflow": {
                "tracking_uri": os.getenv('MLFLOW_TRACKING_URI'),
                "experiment_name": "default_experiment",
                "model_registry_metric_threshold": 0.7,
                "mlrun_path": "mlruns",
                "backend_store_uri": "mlruns",
                "model_info_path": "model_info",
                "artifact_location": "artifacts",
                "server_config": {"host": "0.0.0.0", "port": "5000"}
            },
            "project": {"dataset_name": "default", "model_name": "default"},
            "dataset": {"default": {}},
            "models": {"default": {}},
            "common": {
                "checkpoint": {
                    "filename": "model-{epoch:02d}-{val_loss:.2f}",
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 3,
                    "save_last": True
                }
            },
            "hpo": {}
        }

    def _setup_paths(self):
        """경로 설정"""
        base_path = Path("/tmp" if self.is_render else self.project_root)
        self.paths = {
            'data': base_path / 'data',
            'raw_data': base_path / 'data' / 'raw',
            'processed_data': base_path / 'data' / 'processed',
            'models': base_path / 'models',
            'logs': base_path / 'logs'
        }

    def _setup_mlflow(self):
        """MLflow 설정"""
        mlflow_config = self._config.get("mlflow", {})
        # Render의 MLflow 서버 URL 설정
        tracking_uri = "https://upstageailab-ml-pjt-ml-p4-for-deploy.onrender.com"
        
        self.mlflow = MLflowConfig(
            tracking_uri=tracking_uri,
            experiment_name=mlflow_config.get("experiment_name", "default"),
            model_registry_metric_threshold=mlflow_config.get("model_registry_metric_threshold", 0.7),
            mlrun_path=self.paths['logs'] / "mlruns",
            backend_store_uri=self.paths['logs'] / "mlruns",
            model_info_path=self.paths['models'] / "model_info",
            artifact_location=self.paths['models'] / "artifacts",
            server_config=mlflow_config.get("server_config", {})
        )

    def _setup_project_config(self):
        """프로젝트 관련 설정"""
        self.project = self._config.get("project", {})
        self.dataset = self._config.get("dataset", {})
        self.data = self.dataset.get(self.project.get("dataset_name", "default"), {})
        self.models = self._config.get("models", {})
        self.model_config = self.models.get(self.project.get("model_name", "default"), {})
        self.common = self._config.get("common", {})
        self.hpo = self._config.get("hpo", {})

    def _find_project_root(self) -> Path:
        """프로젝트 루트 디렉토리 찾기"""
        if self.is_render:
            return Path(os.getenv('RENDER_PROJECT_ROOT', '/opt/render/project/src'))
        
        current_dir = Path(__file__).resolve().parent
        while current_dir.name:
            if (current_dir / 'src').exists() or (current_dir / 'config' / 'config.yaml').exists():
                return current_dir
            current_dir = current_dir.parent
        raise RuntimeError("Project root directory not found")

    def _create_directories(self):
        """필요한 디렉토리 생성"""
        # 기본 경로 생성
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # MLflow 관련 디렉토리 생성
        if hasattr(self, 'mlflow'):
            mlflow_paths = [
                self.mlflow.mlrun_path,
                self.mlflow.backend_store_uri,
                self.mlflow.artifact_location,
            ]
            for path in mlflow_paths:
                Path(path).mkdir(parents=True, exist_ok=True)
            
            # model_info 디렉토리 생성
            Path(self.mlflow.model_info_path).parent.mkdir(parents=True, exist_ok=True)

    def _check_required_env_vars(self):
        """필수 환경 변수 확인"""
        required_vars = ['MLFLOW_TRACKING_URI']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")