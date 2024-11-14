import os
import argparse
import shutil

def find_sequence_directories(data_root, tau, dt):
    sets = ['train', 'val', 'test']  # train, val, test のディレクトリを確認
    found_dirs = []  # 存在するディレクトリを格納するリスト
    
    for data_set in sets:
        set_path = os.path.join(data_root, data_set)
        
        # train, val, test の各ディレクトリ内にあるシーケンスディレクトリを取得
        if os.path.isdir(set_path):
            sequence_dirs = [name for name in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, name))]
            
            for seq_dir in sequence_dirs:
                tau_dt_path = os.path.join(set_path, seq_dir, f'tau={tau}_dt={dt}')
                
                # tau, dt ディレクトリが存在するか確認
                if os.path.isdir(tau_dt_path):
                    found_dirs.append(tau_dt_path)
        else:
            print(f"ディレクトリが存在しません: {set_path}")
    
    return found_dirs

def delete_directories(directories):
    for directory in directories:
        shutil.rmtree(directory)
        print(f"削除しました: {directory}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定されたTauとDtのディレクトリを探索および削除するスクリプト")
    parser.add_argument("--data_root", type=str, required=True, help="データルートのパス")
    parser.add_argument("--tau", type=int, required=True, help="tauの値")
    parser.add_argument("--dt", type=int, required=True, help="dtの値")
    
    args = parser.parse_args()
    
    # 指定のディレクトリを探索
    found_dirs = find_sequence_directories(args.data_root, args.tau, args.dt)
    
    if found_dirs:
        print("以下のディレクトリが存在します:")
        for dir_path in found_dirs:
            print(f"  - {dir_path}")
        
        # 削除実行の確認
        confirm = input("これらのディレクトリを削除しますか？ (Y/n): ").strip().lower()
        if confirm == 'y':
            delete_directories(found_dirs)
            print("すべての指定ディレクトリを削除しました。")
        else:
            print("削除をキャンセルしました。")
    else:
        print("指定のディレクトリは存在しませんでした。")
