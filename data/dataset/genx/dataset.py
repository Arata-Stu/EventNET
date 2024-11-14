import os
from torch.utils.data import ConcatDataset
from .sequence import SequenceDataset  # SequenceDataset クラスをインポート

class PropheseeConcatDataset(ConcatDataset):
    def __init__(self, base_data_dir: str, mode: str, tau: int, delta_t: int, 
                 sequence_length: int = 1, guarantee_label: bool = False, transform=None):
        self.base_data_dir = base_data_dir
        self.sequence_length = sequence_length
        self.guarantee_label = guarantee_label
        self.mode = mode
        self.tau = tau
        self.delta_t = delta_t

        # tau と delta_t に対応するパスを生成
        tau_delta_dir = f"tau={self.tau}_dt={self.delta_t}"
        full_tau_delta_path = os.path.join(self.base_data_dir, tau_delta_dir)

        if not os.path.isdir(full_tau_delta_path):
            raise ValueError(f"The directory for tau and delta_t '{tau_delta_dir}' does not exist in {self.base_data_dir}")

        # 指定されたモードのディレクトリから全シーケンスディレクトリを取得
        mode_dir = os.path.join(full_tau_delta_path, self.mode)
        if not os.path.isdir(mode_dir):
            raise ValueError(f"The directory for mode '{self.mode}' does not exist in {full_tau_delta_path}")
        
        # 各シーケンスディレクトリを確認し、SequenceDataset インスタンスを生成してリストに追加
        datasets = []
        for sequence in os.listdir(mode_dir):
            sequence_path = os.path.join(mode_dir, sequence)
            if os.path.isdir(sequence_path):
                datasets.append(
                    SequenceDataset(
                        data_dir=sequence_path,
                        mode=self.mode,
                        sequence_length=self.sequence_length,
                        guarantee_label=self.guarantee_label,
                        transform=transform,
                    )
                )
        
        # ConcatDataset の初期化を利用して複数のデータセットを結合
        super().__init__(datasets)

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError("index out of range")
