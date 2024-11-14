import os
import argparse

def check_sequence_directories(data_root, tau, delta):
    sets = ['train', 'val', 'test']  # train, val, test のディレクトリを確認
    missing_dirs = []  # 存在しないディレクトリを格納するリスト
    
    for data_set in sets:
        set_path = os.path.join(data_root, data_set)
        
        # train, val, test の各ディレクトリ内にあるシーケンスディレクトリを取得
        if os.path.isdir(set_path):
            sequence_dirs = [name for name in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, name))]
            
            for seq_dir in sequence_dirs:
                tau_delta_path = os.path.join(set_path, seq_dir, f'tau={tau}_dt={delta}')
                
                # tau, delta ディレクトリが存在するか確認
                if not os.path.isdir(tau_delta_path):
                    missing_dirs.append(tau_delta_path)
        else:
            print(f"ディレクトリが存在しません: {set_path}")
    
    # 存在しないディレクトリがある場合に出力
    if missing_dirs:
        print("存在しないディレクトリがあります:")
        for missing in missing_dirs:
            print(f"  - {missing}")
    else:
        print("すべてのディレクトリが存在します。")
    
    return missing_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定されたTauとDeltaのディレクトリを探索するスクリプト")
    parser.add_argument("--data_root", type=str, required=True, help="データルートのパス")
    parser.add_argument("--tau", type=int, required=True, help="tauの値")
    parser.add_argument("--dt", type=int, required=True, help="deltaの値")
    
    args = parser.parse_args()
    
    # 引数を使って関数を呼び出し
    missing_dirs = check_sequence_directories(args.data_root, args.tau, args.dt)
