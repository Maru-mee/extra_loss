import csv
import os

def log_gradient(record):
    """
    受け取ったデータをCSVに書き込む。step=0の場合はファイルをリセットする。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "grad_log.csv")
    
    # stepが0なら上書きモード(w)、それ以外は追記モード(a)
    mode = 'w' if record.get("step") == 0 else 'a'
    
    try:
        # 新規作成時または上書き時はヘッダーが必要
        write_header = not os.path.isfile(file_path) or mode == 'w'
        
        with open(file_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)
    except Exception:
        pass