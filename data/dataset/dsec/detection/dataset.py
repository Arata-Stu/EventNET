import os
import yaml
from torch.utils.data import ConcatDataset
from .sequence import SequenceDataset

import os
import yaml
from torch.utils.data import ConcatDataset

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
        
        # tau と delta_t に対応するサブディレクトリを含む SequenceDataset を作成
        datasets = []
        for sequence in split_sequences:
            sequence_path = os.path.join(self.base_data_dir, sequence)
            tau_delta_dir = f"tau={self.tau}_dt={self.delta_t}"
            full_path = os.path.join(sequence_path, tau_delta_dir)
                
            if os.path.isdir(full_path):
                datasets.append(
                    SequenceDataset(
                        data_dir=full_path,
                        sequence_length=self.sequence_length,
                        guarantee_label=self.guarantee_label,
                        transform=transform,
                    )
                )
        
        # ConcatDataset の初期化を利用して複数のデータセットを結合
        super().__init__(datasets)