import os
import argparse
import shutil

def check_existing_directories(target_dir, tau, dt):
    sets = ['train', 'val', 'test']
    existing_count = 0  # 存在するディレクトリのカウント
    
    for data_set in sets:
        set_path = os.path.join(target_dir, data_set)
        if os.path.isdir(set_path):
            sequence_dirs = [name for name in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, name))]
            for seq_dir in sequence_dirs:
                tau_dt_path = os.path.join(set_path, seq_dir, f'tau={tau}_dt={dt}')
                if os.path.isdir(tau_dt_path):
                    existing_count += 1
    
    return existing_count

def copy_directory(source_path, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)  # 必要なディレクトリ構造を作成
    shutil.copytree(source_path, target_path)
    print(f"コピーしました: {source_path} -> {target_path}")

def find_and_copy_directories(source_dirs, target_dir, tau, dt):
    sets = ['train', 'val', 'test']  # train, val, test のディレクトリを確認
    copy_jobs = []  # コピーすべきディレクトリのリスト
    
    # ターゲットディレクトリに既に存在する指定されたtau, dtディレクトリを確認
    existing_count = check_existing_directories(target_dir, tau, dt)
    if existing_count > 0:
        print(f"ターゲットディレクトリには既に {existing_count} 個の指定されたディレクトリが存在します。")
    else:
        print("ターゲットディレクトリには指定されたディレクトリは存在しません。")

    # 各ソースディレクトリに対してコピー対象を確認
    for source_root in source_dirs:
        for data_set in sets:
            set_path = os.path.join(source_root, data_set)
            
            if os.path.isdir(set_path):
                sequence_dirs = [name for name in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, name))]
                
                for seq_dir in sequence_dirs:
                    source_tau_dt_path = os.path.join(set_path, seq_dir, f'tau={tau}_dt={dt}')
                    target_tau_dt_path = os.path.join(target_dir, data_set, seq_dir, f'tau={tau}_dt={dt}')
                    
                    # tau, dt ディレクトリが存在するか確認し、コピーリストに追加
                    if os.path.isdir(source_tau_dt_path):
                        copy_jobs.append((source_tau_dt_path, target_tau_dt_path))
    
    # コピー実行の確認
    if copy_jobs:
        print(f"\nコピー対象のディレクトリは合計 {len(copy_jobs)} 個です。")
        confirm = input("これらのディレクトリをコピーしますか？ (Y/n): ").strip().lower()
        if confirm == 'y':
            for source, target in copy_jobs:
                copy_directory(source, target)
            print("すべての指定ディレクトリをコピーしました。")
        else:
            print("コピーをキャンセルしました。")
    else:
        print("コピー対象のディレクトリは見つかりませんでした。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="複数のソースからターゲットディレクトリにデータをコピーするスクリプト")
    parser.add_argument("--source_dirs", type=str, nargs='+', required=True, help="ソースディレクトリのパス (複数指定可能)")
    parser.add_argument("--target_dir", type=str, required=True, help="ターゲットディレクトリのパス")
    parser.add_argument("--tau", type=int, required=True, help="tauの値")
    parser.add_argument("--dt", type=int, required=True, help="dtの値")
    
    args = parser.parse_args()
    
    # ディレクトリをコピー
    find_and_copy_directories(args.source_dirs, args.target_dir, args.tau, args.dt)
