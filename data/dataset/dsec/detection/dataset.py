import os
import yaml
from torch.utils.data import ConcatDataset
from .sequence import SequenceDataset

class DSECConcatDataset(ConcatDataset):
    def __init__(self, base_data_dir: str, mode: str, tau: int, delta_t: int, 
                 sequence_length: int = 1, guarantee_label: bool = False, transform=None, config_path: str = None):
        """
        Args:
            base_data_dir (str): ベースのデータディレクトリのパス。
            mode (str): 'train', 'val', 'test' のいずれか。
            tau (int): タウの値。
            delta_t (int): デルタtの値。
            sequence_length (int): シーケンスの長さ。
            guarantee_label (bool): True の場合、ラベルが存在するシーケンスのみを含める。
            transform (callable, optional): データに適用する変換関数。
            config_path (str): 分割を定義したYAMLファイルのパス。
        """
        self.base_data_dir = base_data_dir
        self.sequence_length = sequence_length
        self.guarantee_label = guarantee_label
        self.mode = mode
        self.tau = tau
        self.delta_t = delta_t

        # YAMLファイルから分割の設定を読み込む
        if config_path is None:
            raise ValueError("config_path is required")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        split_sequences = config['splits'].get(mode, [])
        
        # tau と delta_t の組み合わせディレクトリを取得
        tau_delta_dir = f"tau={self.tau}_dt={self.delta_t}"
        tau_delta_path = os.path.join(self.base_data_dir, tau_delta_dir)

        if not os.path.isdir(tau_delta_path):
            raise ValueError(f"The directory for tau={self.tau} and delta_t={self.delta_t} does not exist in {self.base_data_dir}")
        
        # tau_delta_path の下にある各シーケンスに対応する `SequenceDataset` を生成
        datasets = []
        for sequence in split_sequences:
            sequence_path = os.path.join(tau_delta_path, sequence)
                
            if os.path.isdir(sequence_path):
                datasets.append(
                    SequenceDataset(
                        data_dir=sequence_path,
                        sequence_length=self.sequence_length,
                        guarantee_label=self.guarantee_label,
                        transform=transform,
                    )
                )
        
        # ConcatDataset の初期化を利用して複数のデータセットを結合
        super().__init__(datasets)
