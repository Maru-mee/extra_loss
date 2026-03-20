# -------------------------------------------
# loss_extra_calc
# 画像生成AIにおいて、様々なlossの統計結果を生成します
# -------------------------------------------

is_print_screen         = True  # Falseにより、一切の画面表示をOFF
print_interval_step     = 50    # print表示のstep間隔。

is_debug_mode           = False
is_debug_mode_grad      = False
is_debug_mode_PCgrad    = False

import collections
import math
import numpy as np
import random
import torch
import torch.nn.functional
from torchvision.transforms import RandomCrop

import library.train_util as train_util
from library.custom_train_functions import (
    apply_masked_loss,
)

_current_snr_weight = None
_current_mask = None
_random_seed_1 = 0

_print_storage = []
def print_storage(mode, content=None):
    """
    print文をまとめて出力することで、print呼び出しによるオーバーヘッドを減らす関数
    mode = keepで溜めて、mode = printで溜めた文を一気に表示
    """
    if mode == "keep":
        _print_storage.append(str(content))
    elif mode == "print":
        if _print_storage:
            print("\n".join(_print_storage))
            _print_storage.clear()

def get_image_hw(image_tensor):
    """
    Tensorから画像の高さ(H)と幅(W)を取得する。
    画像がバッチ化されているかに関わらず動作する。
    """
    if image_tensor.dim() == 4: # batchの場合
        _, _, H, W = image_tensor.shape
    elif image_tensor.dim() == 3: # batch出ない場合（=画像１つだけの場合）
        _, H, W = image_tensor.shape
    else:
        raise ValueError("Unsupported tensor dimensions. Expected 3 or 4 dimensions.")
        
    area_latents = H * W
    return H, W, area_latents


# ==============================================================================================

        
#def cal_loss_edge_dist(target, noise_pred, args, huber_c):
    # 削除 2026/3/9

#def cal_loss_edge_match(target, noise_pred, args, huber_c):
    # 削除 2026/3/9  
    
#def cal_loss_grad(target, noise_pred, args, huber_c):
    # 削除 2026/3/9

#_fft_cache = {} 
#def cal_loss_fft_amp(target, noise_pred, args, huber_c, fft_slice_count):
    # 削除 2026/3/9
    
#def cal_loss_var(target, noise_pred, args, huber_c, var_method="none"):
    # 削除 2026/3/9
       
#def cal_loss_rgb_bias(target, noise_pred, args, huber_c):
    # 削除 2026/3/9

def calc_loss_pool(target, noise_pred, args, huber_c, reso_scale, is_above_limit, pool_num):
    """
    画像単体のpool分割したうえで、それぞれの領域を比較する。
    これがあることで、画像単体としてのバランスや、人物の基本骨格が取れるようになる
    骨格が一致しなければ、あらゆる詳細学習が進まない
    しかし、latentsにおけるmean比較というのは茶色くなりがちなので、強い強度での適用は控えたほうがいい
    """
    dtype = target.dtype
    device = target.device    
    
    if not pool_num == 1: # pool_num=1を特別に許可する。平均色の学習としてはどんな解像度でも有意義
        if not is_above_limit:  # 解像度が低い場合、信頼性が著しく低下する
            return torch.zeros(1, device=device, dtype=dtype)

    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)

    def extract_features(x, pool_num):
        # 空間情報の抽出：統計量を測定し、特徴を際立たせる
        
        # (B, C, H, W) -> (B, C, H*W) : チャンネルごとの画素平坦化
        x_flat = x.flatten(2)
        device = x.device

        features = []    
    
        pool_x  = torch.nn.functional.adaptive_avg_pool2d(x, (pool_num, pool_num))
        pool_x = torch.nn.functional.normalize(pool_x.float(), p=2, dim=1, eps=1e-8)
                             
        features = [
            pool_x.flatten(1), 
        ]
        
        return torch.cat(features, dim=1)
    
    # 特徴抽出と標準化を一括処理
    pool_pred   = extract_features(pred_latents, pool_num)
    pool_target = extract_features(target_latents, pool_num)
    
    boost = 0.01
    scales = reso_scale * boost
    
    pool_pred.mul_(scales)
    pool_target.mul_(scales)
    
    loss = train_util.conditional_loss(
        pool_pred.float(),
        pool_target.float(),
        reduction="none",
        loss_type=args.loss_type,
        huber_c=huber_c
    )
    
    return loss
    
def calc_loss_ch_cosine(target, noise_pred, args, huber_c, reso_scale):
    """
    ピクセル単位のチャネル間ベクトル方向の一致度を算出。
    pixel perfectやドット検出（多少ではあるがエッジ検出）の効果があり、新しい画像要素を発見するのに特に有効。
    どのtimestepでもそこそこの効果を持つ、MSE同等以上の使い勝手を持つ
    """
    eps = 1e-8
    
    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)

    # チャンネル方向(dim=1)で正規化
    target_norm = torch.nn.functional.normalize(target_latents.float(), p=2, dim=1, eps=eps)
    pred_norm = torch.nn.functional.normalize(pred_latents.float(), p=2, dim=1, eps=eps)
    
    #target_norm.mul_(reso_scale)
    #pred_norm.mul_(reso_scale)

    # 損失計算
    loss = train_util.conditional_loss(
        pred_norm.float(),
        target_norm.float(),
        reduction="none",
        loss_type=args.loss_type,
        huber_c=huber_c
    )
    
    # 元の次元数に戻す
    if not is_batched:
        loss = loss.squeeze(0)
        
    return loss
    
#def calc_loss_ch_flow_1(target, noise_pred, args, huber_c, reso_scale, is_above_limit):
    # 削除 2026/3/9

def calc_loss_ch_flow_2(target, noise_pred, args, huber_c, reso_scale, is_above_limit):
    """
    連続座標サンプリングによるベクトル相関を全方位・等距離で同期。
    loss_ch_cosineで低下するピクセル間の連続性（ケロイドなどの学習の副産物）を抑制する
    真円状のエッジ検出に優れている。ピクセル単位ではなく、該当距離（ピクセルの隙間含む）の値を滑らかに取るためノイズに強い
    """
    dtype = target.dtype
    device = target.device
    eps = 1e-8

    if not is_above_limit:
        # 解像度が低い場合、計算困難な境界影響が強くなり、境界クロップが発生しやすくなる
        return torch.zeros(1, device=device, dtype=dtype)

    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)

    def create_base_grid(B, H, W):
        """1. 基準となる座標の網を作る"""
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )
        return torch.stack([grid_x, grid_y], dim=-1).to(dtype).unsqueeze(0).expand(B, -1, -1, -1)

    def sample_by_angle(latents, base_grid, angle, r, step_h, step_w):
        """2. 指定した角度に網をずらして値を吸い出す"""
        offset_x = math.cos(angle) * r * step_w
        offset_y = math.sin(angle) * r * step_h
        
        sampling_grid = base_grid.to(latents.dtype).clone()
        sampling_grid[..., 0] += offset_x
        sampling_grid[..., 1] += offset_y
        
        sampled_latents = torch.nn.functional.grid_sample(
            latents, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        # 補足：padding_mode='zeros'の場合、画像端にある被写体が「黒い壁」と隣接していると判定され、そこに実在しない強烈なコントラスト（偽のエッジ）が発生するリスク有り
        
        return sampled_latents, sampling_grid

    def compute_weighted_diff(orig_latents, sampled_latents, grid):
        """3. ターゲットの差分を重みとして、有効領域のみ抽出する（画像の外を対象外とする）"""
        mask = (grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)
        valid_indices = mask.unsqueeze(1).expand_as(orig_latents)
        
        weight = torch.abs(orig_latents - sampled_latents)
        diff = (orig_latents - sampled_latents) * weight
        
        return diff[valid_indices]

    def get_ch_flow(target, pred):
        H, W, _ = get_image_hw(target)
        B, C, _, _ = target.shape
        base_grid = create_base_grid(B, H, W)
        
        # 検索半径
        r = 2.0
        step_h, step_w = 2.0 / (H - 1), 2.0 / (W - 1)
        angles = torch.linspace(0, 2 * math.pi, 9, device=device)[:-1]
        
        t_list, p_list = [], []
        for angle in angles:
            # 比較対象となるピクセルの値を抽出
            sampled_t, grid = sample_by_angle(target, base_grid, angle.item(), r, step_h, step_w)
            sampled_p, _    = sample_by_angle(pred, base_grid, angle.item(), r, step_h, step_w)
            
            # 基準と比較対象との差分を計算
            t_list.append(compute_weighted_diff(target, sampled_t, grid))
            p_list.append(compute_weighted_diff(pred, sampled_p, grid))
            
        return torch.cat(t_list), torch.cat(p_list)

    target_flows_flat, pred_flows_flat = get_ch_flow(target_latents, pred_latents)
    
    target_flows_flat.mul_(reso_scale)
    pred_flows_flat.mul_(reso_scale)

    loss = train_util.conditional_loss(
        pred_flows_flat.float(), 
        target_flows_flat.float(),
        reduction="none", 
        loss_type=args.loss_type, 
        huber_c=huber_c
    )
    
    
    return loss

def calc_loss_pair_correlation(target, noise_pred, args, huber_c, is_above_limit, scale_px):
    """
    地点間の相関（Self-Correlation）による構造損失。
    1枚の画像内の全座標をペアにして関係性を網羅（総当たり方式）することで、
    特定の部位（顔など）への依存を排し、画像全体の空間的な秩序を強制的に学習させる。
    
    scale_px : 元画像サイズスケールでの、１個あたりのgridサイズ
    poolの分割数
    """
    dtype = target.dtype
    device = target.device

    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)

    def apply_distance_weight(i, j, num_grid_w, dtype, device):
        """
        グリッド上の距離に応じて重みを生成
        1マス隣: 1.5倍 / その他: 1.0倍に近い値へ減衰
        """
        # インデックスから2次元座標(x, y)を復元
        dx = (i % num_grid_w - j % num_grid_w).abs().to(dtype)
        dy = (i // num_grid_w - j // num_grid_w).abs().to(dtype)
        
        # 直線距離(ユークリッド距離)に基づいて算出
        dist = torch.sqrt(dx**2 + dy**2)

        # 1.0(ベース) + 強度 * 指数減衰
        # dist=1.0 のとき 1.5 になるよう設定
        strength = 0.5
        decay = 0.8 
        weights = 1.0 + strength * torch.exp(-decay * (dist - 1.0))
        
        return weights

    # サイズから分割数決定
    H, W, _ = get_image_hw(target)
    scale_latents = scale_px / 8 # px → latents
    num_grid_h = int(max(1, H // scale_latents))
    num_grid_w = int(max(1, W // scale_latents))
    
    # 4x4を確保できないならば精度不足。120通りのペア数を保証することで多様性を確保
    if num_grid_h < 4 or num_grid_w < 4:
        return torch.zeros(1, device=device, dtype=dtype)

    # 空間情報の抽出
    target_small = torch.nn.functional.adaptive_avg_pool2d(target_latents.float(), (num_grid_h, num_grid_w))
    pred_small = torch.nn.functional.adaptive_avg_pool2d(pred_latents.float(), (num_grid_h, num_grid_w))
    
    # 特徴ベクトル化 (B, HW, C)
    target_feat = target_small.flatten(2).transpose(1, 2)
    pred_feat = pred_small.flatten(2).transpose(1, 2)
    
    # 相関行列の生成（要素ごとの差分で番地情報を維持）
    # (B, HW, 1, C) - (B, 1, HW, C) -> (B, HW, HW, C)
    target_corr = target_feat.unsqueeze(2) - target_feat.unsqueeze(1)
    pred_corr = pred_feat.unsqueeze(2) - pred_feat.unsqueeze(1)
    
    num_locations = num_grid_h * num_grid_w

    # 重複と自己相関を排除するインデックスを抽出
    indices = torch.triu_indices(num_locations, num_locations, offset=1, device=device)
    i, j = indices[0], indices[1]  # (i:基準点, j:比較点) 
    
    # チャンネル次元(dim=-1)を保持したままペアを抽出
    # (B, num_pairs, C)
    target_pairs = target_corr[:, i, j, :]
    pred_pairs   = pred_corr[:, i, j, :]
        
    # 距離重みの適用
    dist_weight = apply_distance_weight(i, j, num_grid_w, dtype, device).unsqueeze(-1)
    target_pairs = target_pairs * dist_weight
    pred_pairs   = pred_pairs * dist_weight
    
    # 補正
    boost = 1.0
    target_pairs *= boost
    pred_pairs   *= boost

    loss = train_util.conditional_loss(
        pred_pairs, 
        target_pairs,
        reduction="none", 
        loss_type="l2",  # ほかの関数同様にargs.loss_typeすると差分が検出できず、grad.abs.max = grad.abs.meanになってしまうので、仕方なくL2
        huber_c=huber_c
    )
            
    return loss
    
def calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit, mode, pool_num=1):
    """
    ・バッチ内の画像間の類似度構造（Relation）をターゲットと同期させることで、
      各画像が持つ固有の特徴的な差異を学習し、トークンの定着率を高める。
    ・特定の画像の学習に偏った学習をさせない。メジャーなプロンプトへ収束してしまう現象を抑制
    ・UNet学習済みの画像パターンA,パターンBに無理やり当てはめるような「手抜き生成」を防ぎ、
      入力トークンごとの微細な描き分けをモデルに強制させる。
    ・テンプレ的な「AIらしさ」や「平均的な綺麗さ」は減るが、プロンプトへの忠実度は向上する。
    """
    dtype = target.dtype
    device = target.device    
    
    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)
    batch_size = target_latents.shape[0]     
    
    is_execute_flag = True
    
    if batch_size < 2:
        # batch_size = 1なら計算する目的が消滅する
        is_execute_flag = False
    
    if not is_above_limit:
        # 解像度が低い場合、いくつかの統計値に対する信頼性が著しく低下する可能性がある
        
        if mode == "pool" and pool_num == 1:
            # この条件ならば、平均色をbatch比較する意義があるので、skipしたくない
            pass
            
        elif mode=="ch_cosine":
            # ch_cosineモードであれば、ピクセル同士の比較であるため、解像度に関係なく使用可能
            pass
        else:
            is_execute_flag = False       
     
    if not is_execute_flag:
        return torch.zeros(1, device=device, dtype=dtype)
        
    def extract_features(x, mode):
        # 空間情報の抽出：統計量を測定し、特徴を際立たせる
        
        # (B, C, H, W) -> (B, C, H*W) : チャンネルごとの画素平坦化
        x_flat = x.flatten(2)
        device = x.device
        dtype  = x.dtype

        features = []
                       
        if mode=="tones":
            # 輝度の階層別比較。
            # 位置づけは、stdの上位互換
            # 平均色はtonesでは考慮せず、平均色からの差分だけを評価する。
            # 補足：平均色はpoolが担当して、重複カウントを抑制。この重複を抑制しないと、明確に色がおかしくなり、めちゃくちゃになる
            # tones の目的は、分布の「広がり（コントラスト/彩度）」を合わせることであり、明るさの絶対位置を合わせなくても破綻しない
            
            x_delta = x_flat - x_flat.mean(dim=2, keepdim=True) # 平均を除去
            sampling_reso = 10
            
            q_list = torch.linspace(0.0, 1.0, sampling_reso, device=device, dtype=torch.float32)
            tones = torch.quantile(x_delta.float(), q_list, dim=2).permute(1, 2, 0).flatten(1) 
            features.append(tones)

            num_features = sampling_reso 
            boost = 1.0  #経験則からこのくらい。
            #print(f"num_features(tones)\t{num_features}")
            
        elif mode=="pool":
            # 各領域の平均値を比較する
            # 注：    輝度meanはここで評価されているので、ほかのloss_batchでは比較するべきではない            
            #        7分割のような細かな分割設計にしたい場合は、必ず3分割などを中継して、レイアウトを学ばせること。
            #       3x3だけだと物足りないように感じるかもしれないが、ほとんどの画像で3x3を学べれば、実質細かい部分もわかる
            #       5x5を追加することで、高周波側へ逃がしてしまう力を弱める。
            #       7x7以上はスライス跡が目立つのでやめるべき。縦長or横長では分割困難になる
            
            pool_x = torch.nn.functional.adaptive_avg_pool2d(x, (pool_num, pool_num))

            # grad過大にならないように正規化
            pool_x = torch.nn.functional.normalize(pool_x.float(), p=2, dim=1, eps=1e-8)     
                                 
            features = [
                pool_x.flatten(1), 
            ]
            
            num_features    = 1
            boost           = 0.01   # 体感上このくらいがbaseと同程度のgradになる
            
        elif mode=="ch_cosine":
            # ch_cosineと同等
            # poolよりもピクセル単位で細かく比較できるので、キャプション差を捉えやすい（たとえば、人の顔は狭い区間にたくさんのタグがあるため）
            
            # そのため、平均輝度のみを扱うbatch_tonesの役割を数学的に内包し、上位互換として機能する。
            # 理由は、ch_cosineはチャンネル間の比率を固定するため、比率が確定すれば輝度（合計値）の自由度も制限される。
            # timestepがノイズだらけのときでも使用できるのがアドバンテージ
                            
            B, C, H, W = x.shape

            x_norm = torch.nn.functional.normalize(x.float(), p=2, dim=1, eps=1e-8)

            features = [x_norm]
            
            num_features = (C * H * W) / 1.0e+5 # 正規化の都合で信号が弱すぎるので補正。補強しないとアンダーフローしてloss=0になる
            num_features = max(num_features, 1e-8) # 除算時のゼロ除算による巨大化対策
            boost = 0.3
            
        elif mode=="others": 
            # 基本的には使わない、過去の統計値
            #mean = torch.mean(x, dim=(2, 3))  # 全体的な色味や明るさのトーン → 画像が茶色くなる原因かもしれない、ボツ
            #amax = top_k_mean(largest=True)   # 最も強い光（ハイライト）の勢い → timestep1000付近で4隅欠損するのでボツ
            #amin = top_k_mean(largest=False)  # 最も深い影（ヌケ）の沈み込み → timestep1000付近で4隅欠損するのでボツ
            #std  = torch.std(x, dim=(2, 3))   # 描き込みの密度や質感の激しさ → tonesが上位互換となるのでボツ
            features.extend([
                #mean,
                #amax, 
                #amin,
                #std,
            ])
            num_features = 1 # 適当に設定
            boost = 1.0
            
        return torch.cat(features, dim=1), num_features, boost         
    
    # 特徴抽出と標準化を一括処理
    feat_pred, num_features, boost  = extract_features(pred_latents, mode)
    feat_target, _, _               = extract_features(target_latents, mode)
    
    # p=1(L1距離)を採用し、特徴量の差分を定義
    # 類似度（内積方式）はL2特性を持つため、値が過小になり外れ値ばかりが優先される傾向がある。
    # それを避けるため、L1ノルム（vector_norm ord=1）にて各要素の差を忠実に拾う。
    # また、meanではなくsum（または相当する集計）を使用することで、
    # 正負の誤差による相殺・対消滅を防ぎ、バッチ内の関係性を確実に勾配へ乗せる。
    diff_pred   = feat_pred.unsqueeze(1)   - feat_pred.unsqueeze(0)
    diff_target = feat_target.unsqueeze(1) - feat_target.unsqueeze(0)

    # L1ベクトルノルムとして集計し、要素数で正規化することで勾配爆発を抑制
    rel_pred = torch.linalg.vector_norm(diff_pred, ord=1, dim=2) / feat_pred.shape[1]
    rel_target = torch.linalg.vector_norm(diff_target, ord=1, dim=2) / feat_target.shape[1]

    # 数値的安定性のための下限保証（アンダーフローによる勾配消失対策）
    # 注.clmapで置き換えてしまうと勾配追跡が消失する
    rel_pred    = rel_pred + 1e-10
    rel_target  = rel_target + 1e-10
    
    # 5. 対角成分（自己相関で０になってしまう無価値な要素）の除外
    mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
    rel_pred    = rel_pred[mask]
    rel_target  = rel_target[mask]
    
    # 補正。
    # これがないと値が小さすぎて、使い物にならない
    # ここで掛けるブーストは、合算時の重み倍率とは違い、L2対象をL1へ引き上げる効果も生まれる(ハズレ以外も学べる)
    # トークン分離能力を高めるには、重み倍率1.0では到底足りないか？
    # 260303 boost 3.0→10.0 トークンを学ぶ力が足りないので、増加。10は大きすぎ、1は小さすぎ
    scales = boost
    rel_pred.mul_(scales)
    rel_target.mul_(scales)

    loss = train_util.conditional_loss(
        rel_pred.float(),
        rel_target.float(),
        reduction="none",
        loss_type=args.loss_type,
        huber_c=huber_c
    )    
    
    return loss
    
#-----------------------------------------

_LOSS_CONFIG = {
    #名称と、(重み倍率, gamma, deadband)
    # gamma     ：lossを何乗するか。大きいほど学習後期よりも学習序盤に寄与する。
    # deadband  ：この閾値以下のlossをカットして、過適合を防ぐ
    "base   ":  (1.0, 1.0, 0.0),    # 最も大切なlossではあるが、grad/loss効率が低いので、強調したいところ
    "pool_1x": (0.5, 1.0, 0.0),
    "pool_3x": (0.5, 1.0, 0.0),
    "pool_5x": (0.5, 1.0, 0.0),    
    "ch_cosine": (0.5, 1.0, 0.01),
    "ch_flow":  (0.5, 1.0, 0.01),
    "pair_128px": (0.5, 1.0, 0.0),
    "pair_64px": (0.5, 1.0, 0.0),
    "pair_32px": (0.5, 1.0, 0.0),
    "batch_p_1x": (0.5, 1.0, 0.0), 
    "batch_p_3x": (0.5, 1.0, 0.0),
    "batch_p_5x": (0.5, 1.0, 0.0),    
    "batch_cos": (0.5, 1.0, 0.01), 
}

_LOSS_NAMES = list(_LOSS_CONFIG.keys())

    
def combine_losses_dynamically(
    losses_list: list[torch.Tensor], 
    global_step,
):
    if not losses_list:
        return torch.tensor(0.0, device='cpu')
    
    # 初期化行程。loss_listの状態を把握したり、変数の初期化を行う --------------------------------
    
    # first_valid_loss : 基本情報取得や初期化のために、Noneではないlossを見つける
    first_valid_loss = next(l for l in losses_list if l is not None)
    first_valid_loss_tensor = first_valid_loss[0] if isinstance(first_valid_loss, tuple) else first_valid_loss
    all_loss = torch.zeros_like(first_valid_loss_tensor)
    
    device = first_valid_loss_tensor.device 
        
    grads_list = [] # 各lossから算出したgrad
    base_shape_tensor = None # 基準となる形状（通常は最初の有効なロス）を特定するための準備
    scalar_only_sum = torch.tensor(0.0, device=device) # PCgradで合成できない、たまに発生するスカラー
    
    
    # 各損失の勾配算出とリスト化 ------------------------------------------------------------
    for i, item in enumerate(losses_list):
        if i < len(_LOSS_NAMES) and item is not None:
            
            loss_value_raw, pred = item
            loss_name = _LOSS_NAMES[i]
            
            # 各lossの個別整形 -----------------------------------------------
            
            static_weight, gamma_value, deadband = _LOSS_CONFIG.get(loss_name, (1.0, 1.0, 0.0))
            
            # 微分対象のテンソルに勾配計算を許可する
            if not pred.requires_grad:
                pred.requires_grad_(True)   
            
            if loss_value_raw is None:
                continue
                
            current_loss_mean = loss_value_raw.mean()
            
            if deadband > 0.0:
                loss_value_raw = torch.nn.functional.relu(loss_value_raw - deadband) # deadband未満を切り捨て

            dynamic_weight = 1.0
            base_gamma = 1.0
            
            all_weight = static_weight * dynamic_weight
            all_gamma  = base_gamma * gamma_value            
                        
            # 重みとガンマを適用したスカラー損失の算出
            # 計算過程で勾配が必要なため、オリジナルのテンソルから計算を開始
            loss_instance = loss_value_raw.clone()
            loss_instance **= all_gamma
            loss_instance *= all_weight
            
            loss_scalar = loss_instance.mean()
            
            if global_step % print_interval_step == 0 or is_debug_mode:
                def loss_bar(loss):
                    max_bar = 10  # 0.05刻みで最大0.5 → 10段階
                    capped_loss = min(loss, 0.5)
                    blocks = int(capped_loss / 0.05)
                    bar = "█" * blocks + " " * (max_bar - blocks)
                    return bar
                bar = loss_bar(loss_instance.mean().item())
                
                indent = "\n" if i == 0 else ""
                print_storage("keep", f"{indent} {loss_name} \tgamma:{base_gamma}*{gamma_value}\tSt_wt:{static_weight:.3f} \tDy_wt:{dynamic_weight:.3f} \tloss補正前/補正後\t{current_loss_mean.item():.3f}/{loss_instance.mean().item():.3f}\t|{bar}|")
            
            
            # calculate loss to grad  ---------------------------------------------
            
            if loss_scalar.grad_fn is None:
                # 勾配がない場合はスカラーとして蓄積
                scalar_only_sum += loss_scalar.detach()
                continue
            
            # 指定された評価軸テンソル（pred）で微分を実行
            # 常に同じ「予測値」の座標系で勾配が算出されるため、衝突検知が可能になる
            grad_tuple = torch.autograd.grad(
                loss_scalar, 
                pred, 
                retain_graph=True, 
                allow_unused=True
            )
            if grad_tuple[0] is not None:
                grad = grad_tuple[0].detach()
                
                # grad clipping (緊急時向け)
                if global_step % print_interval_step == 1 or is_debug_mode_grad:
                    print_storage("keep", f" Grad abs_max,mean:\t{grad.abs().max().item():.2e}\t{grad.abs().mean().item():.2e}\t[{loss_name.strip()}]") # for debug
                
                # grad.clamp_(-1e-5, 1e-5) # SDXL向けの緊急時専用ブレーキ

            else:
                # グラフがつながっていない場合は、形状を合わせたゼロ勾配を代入
                grad = torch.zeros_like(pred).detach()
                
            grads_list.append(grad)
            
            if base_shape_tensor is None:
                base_shape_tensor = loss_value_raw

    if not grads_list:
        return torch.tensor(0.0, device=losses_list[0].device, requires_grad=True)

    # Multi-task Learning（MTL）実行前のgradの処理 -------------------
    
    # gradの並び順を定義
    num_losses = len(grads_list)
    indices = list(range(num_losses))
    random.shuffle(indices) # 評価順をシャッフルして、loss並び順の影響を計算結果に与えにくくする
    
    # 相互にgradを操作できるように、形状を揃える
    # 二重ループ外で形状情報を整理し、最大要素数でパディングを適用
    max_numel = max(g.numel() for g in grads_list)
    flat_grads = []
    original_shapes = []
    for g in grads_list:
        original_shapes.append(g.shape)
        g_f = g.reshape(-1)
        
        if g_f.numel() < max_numel:
            g_f = torch.nn.functional.pad(g_f, (0, max_numel - g_f.numel()))
            
        g_f = torch.nan_to_num(g_f, nan=0.0, posinf=0.0, neginf=0.0) # 計算前に nan/inf を 0 に置換（安全装置） 
        flat_grads.append(g_f)

    flat_grads_2 = [g.clone() for g in flat_grads]

    # デバッグ専用統計値
    conflict_count = 0
    total_reduction_norm = 0.0
    original_norms_sum = 0.0

    # PCgradの適用  ---------------------------------------------

    for i in indices:
        gi_flat = flat_grads_2[i]
        
        for j in indices:
            if i == j: continue
            
            gj_flat = flat_grads[j] # 比較対象は事前平坦化済みテンソル
            
            if gi_flat.dtype != gj_flat.dtype:
                gj_flat = gj_flat.to(gi_flat.dtype)

            dot_prod = torch.dot(gi_flat, gj_flat)

            if is_debug_mode_PCgrad:
                before_norm = torch.norm(gi_flat) # 修正前のノルムを記録
            
            if dot_prod < 0:
                # 異方向（干渉対策）
                # 直交成分だけを残し、干渉成分を相殺する
                conflict_count += 1
                
                # mag_sq（勾配の大きさの二乗）が極端に小さい場合に発生する不定形演算を回避                   
                mag_sq = torch.dot(gj_flat, gj_flat)
                if mag_sq > 1e-12:
                    # 補正係数の絶対値を最大1.0に制限し、勾配の爆発を阻止
                    ratio = torch.clamp(dot_prod / mag_sq, min=-1.0, max=1.0)
                    gi_flat = gi_flat - ratio * gj_flat

            elif dot_prod > 0:
                # 同方向（オーバーシュート対策）：
                # 単純加算による「二重の押し込み」を防ぎ、強い方の要求を優先する
                gi_flat = torch.where(
                    torch.abs(gi_flat) > torch.abs(gj_flat),
                    gi_flat,
                    gj_flat
                )
            else:
                # 直交、またはどちらかの勾配が0の場合は干渉も重複もないためスキップ
                pass
                
            if is_debug_mode_PCgrad:                    
                # 修正後のノルムとの差分から「カットされた量」を蓄積
                after_norm = torch.norm(gi_flat)
                total_reduction_norm += abs((before_norm - after_norm).item())
                original_norms_sum += before_norm.item()

        
        # 修正済み平坦化テンソルを格納
        flat_grads_2[i] = gi_flat

    # 平坦化したテンソルを元の形状に復元して edited_grads を作成
    edited_grads = []
    for k, efg in enumerate(flat_grads_2):
        orig_s = original_shapes[k]
        actual_numel = torch.prod(torch.tensor(orig_s)).item()
        edited_grads.append(efg[:actual_numel].view(orig_s))

    def _accumulate_with_shape_match(base_tensor, add_tensor):
        """
        形状の不一致を補正して加算
        """
        if add_tensor is None or add_tensor.numel() == 0:
            return base_tensor

        # スカラー(dim=0)や1次元(dim=1)の勾配を弾く
        if add_tensor.dim() < 2:
            return base_tensor
        
        add_t = add_tensor.clone()
        if add_t.shape != base_tensor.shape:
            # 2Dテンソル (B, C) を (B, C, 1, 1) に展開
            if add_t.dim() == 2:
                add_t  = add_t.unsqueeze(-1).unsqueeze(-1)
            
            # チャンネル数（次元1）の調整
            if add_t.shape[1] != base_tensor.shape[1] and add_t.shape[1] > 0:
                if base_tensor.shape[1] % add_t .shape[1] == 0:
                    repeat_factor = base_tensor.shape[1] // add_t.shape[1]
                    add_t  = add_t.repeat(1, repeat_factor, 1, 1)
                else:
                    print_storage("keep", "skip:_accumulate_with_shape_match: unexpected_case_001") # noise_predで統一しているので起こらないはず
                    return base_tensor 

            # 空間解像度 (H, W) のリサイズ
            if add_t.shape[2:] != base_tensor.shape[2:]:
                if any(t == 0 for t in add_t.shape):
                    print_storage("keep", "skip:_accumulate_with_shape_match: unexpected_case_002")
                    return base_tensor
                
                add_t = torch.nn.functional.interpolate(
                    add_t, 
                    size=base_tensor.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        
        if add_t.shape == base_tensor.shape: # 形状一致を確認
            base_tensor += add_t
            
        return base_tensor

    accumulated_grad = torch.zeros_like(grads_list[0])
    for eg in edited_grads:
        accumulated_grad = _accumulate_with_shape_match(accumulated_grad, eg)
    
    # 最終的なテンソルとしての再構成
    # 呼び出し側の .mean([1, 2, 3]) に対応するため、スカラー化せず 4次元を維持する
    total_loss_tensor = torch.zeros_like(base_shape_tensor)
    
    # 全ロスの値をテンソル形状を保ったまま合算
    for i, item in enumerate(losses_list):
        if i < len(_LOSS_NAMES) and item is not None:
            loss_value_raw, _ = item
            l_val = loss_value_raw.clone()
            # 形状が異なる場合は base_shape_tensor に合わせる
            if l_val.shape != total_loss_tensor.shape:
                l_val = l_val.mean().expand_as(total_loss_tensor)
            
            total_loss_tensor += l_val

    # 勾配の異常値を丸める
    accumulated_grad.nan_to_num_(nan=0.0, posinf=0.1, neginf=-0.1).clamp_(-0.1, 0.1)

    # 勾配を注入するためのアンカー（通常はnoise_pred）を抽出
    grad_anchor = next(item[1] for item in losses_list if item is not None)

    # 勾配のすり替え
    # total_loss_tensorをdetachして元の勾配を切り離し、PCGrad済みの勾配（grad_anchor経由）を合成
    all_loss = total_loss_tensor.detach()
    all_loss += (accumulated_grad * (grad_anchor - grad_anchor.detach())).sum()
    all_loss += scalar_only_sum

    if is_debug_mode_PCgrad:     
        # PCGrad 統計値の計算
        reduction_rate = (total_reduction_norm / (original_norms_sum + 1e-8)) * 100
        print_storage("keep", f" [PCGrad Stats] Conflicts(+-unmatch): {conflict_count} | Grad Cut Rate: {reduction_rate:.2f}%")    
                   
    return all_loss

def get_loss_all(
    loss_base, 
    target, 
    noise_pred, 
    args, 
    huber_c,
):
    
    global _random_seed_1
    # 生成用ランダムseed 
    _random_seed_1 = random.randint(0, 2**32 - 1)
    
    # 縮小処理をする場合の下限面積[元画像サイズベースのpx単位]
    area_lower_limit_img        = 512 ** 2 
    area_lower_limit_latents    = area_lower_limit_img  // (8**2)
    
    H, W, area_latents = get_image_hw(target)
    
    is_above_limit = area_latents >= area_lower_limit_latents

    # 面積スケーリング係数（1024px=128latent を 1.0 とする）
    # 低解像度画像が優位になりすぎてしまうlossに対して影響度を下げる
    reso_scale = min(1.0, math.sqrt((H * W) / (128 * 128)))
    #print(f"reso_scale\t{reso_scale}") for debug
  
    # 各lossの算出=====================================

    dtype = target.dtype
    device = target.device    

    loss_pool_1x = calc_loss_pool(target, noise_pred, args, huber_c, reso_scale, is_above_limit=True, pool_num=1)
    loss_pool_3x = calc_loss_pool(target, noise_pred, args, huber_c, reso_scale, is_above_limit=True, pool_num=3)
    loss_pool_5x = calc_loss_pool(target, noise_pred, args, huber_c, reso_scale, is_above_limit=True, pool_num=5)
    loss_ch_cosine = calc_loss_ch_cosine(target, noise_pred, args, huber_c, reso_scale)
    loss_ch_flow = calc_loss_ch_flow_2(target, noise_pred, args, huber_c, reso_scale, is_above_limit)
    loss_pair_corr_128px = calc_loss_pair_correlation(target, noise_pred, args, huber_c, is_above_limit, scale_px=128)
    loss_pair_corr_64px = calc_loss_pair_correlation(target, noise_pred, args, huber_c, is_above_limit, scale_px=64)
    loss_pair_corr_32px = calc_loss_pair_correlation(target, noise_pred, args, huber_c, is_above_limit, scale_px=32)
    loss_batch_pool_1x = calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit, mode="pool", pool_num=1)
    loss_batch_pool_3x = calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit, mode="pool", pool_num=3)
    loss_batch_pool_5x = calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit, mode="pool", pool_num=5)
    loss_batch_cos = calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit=True, mode="ch_cosine")
   
    # 統合するlossをリスト化する。
    # リストの位置が重要なので、必ず何かを代入すること。統合をスキップしたい場合はNoneを代入する。
    all_computed_losses = [
        loss_base,
        loss_pool_1x,
        loss_pool_3x,
        loss_pool_5x,
        loss_ch_cosine,
        loss_ch_flow * _current_snr_weight,
        loss_pair_corr_128px,
        loss_pair_corr_64px,
        loss_pair_corr_32px,
        loss_batch_pool_1x,
        loss_batch_pool_3x,
        loss_batch_pool_5x,        
        loss_batch_cos,
    ]
    
    # NaN/Inf補正およびnoise_predとのペアリング
    all_computed_losses = [
        (torch.nan_to_num(l, nan=0.0, posinf=1e-4, neginf=0.0), noise_pred) 
        if l is not None else None 
        for l in all_computed_losses
    ]
    
    return all_computed_losses
    
def calc_extra_losses(
    loss, 
    target, 
    noise_pred, 
    args, 
    huber_c, 
    global_step, 
    accelerator, 
    snr_weight_view,
    current_mask=None,
):
    global _current_snr_weight, _current_mask
    _current_snr_weight = snr_weight_view
    _current_mask = current_mask
        
    # loss値の各種計算
    all_computed_losses = get_loss_all(loss, target, noise_pred, args, huber_c)
         
    # ロスの集計と重み付け
    loss = combine_losses_dynamically(all_computed_losses, global_step)

    if is_print_screen:
        print_storage("print")

    # VRAM解放のため、個々の損失テンソルを削除
    for l in all_computed_losses:
        if l is not None:
            del l
    
    # 不要な変数を削除
    del all_computed_losses
    
    return loss
