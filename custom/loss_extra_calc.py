
is_debug_mode           = False
is_debug_mode_PCgrad    = False

import collections
import cv2
import math
import numpy as np
import PIL.ImageEnhance as ImageEnhance
import PIL.Image as Image
import random
import torch
import torch.nn.functional
from torchvision.transforms.functional import to_pil_image, to_tensor, posterize
from torchvision.transforms import RandomCrop

import library.train_util as train_util
from library.custom_train_functions import (
    apply_masked_loss,
)

"""
# loss記録用
import openpyxl
import collections
_loss_history = []
"""
_current_snr_weight = None
_current_mask = None
_random_seed_1 = 0

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

def apply_adjust_hue_pil(img_pil, angle_degrees):
    hsv_img = img_pil.convert('HSV')
    hsv_img_data = list(hsv_img.getdata())                            
    angle_offset_hsv = int((angle_degrees / 360) * 255)
    new_hsv_data = []
    for h, s, v in hsv_img_data:
        new_h = (h + angle_offset_hsv) % 256
        new_hsv_data.append((new_h, s, v))                            
    hsv_img.putdata(new_hsv_data)
    return hsv_img.convert('RGB')


# キャッシュを格納する辞書
_gauss_ker_cache = collections.defaultdict(dict)

def filtering_gaussian(x, dtype, device):
    """
    入力テンソルxにガウシアンフィルタを適用した結果を返します。
    （元の get_gaussian_kernel 関数の機能を変更）

    Args:
        x (torch.Tensor): 入力テンソル（noise_predまたはtarget）。(B, C, H, W)
        dtype (torch.dtype): テンソルのデータ型。
        device (torch.device): テンソルのデバイス。

    Returns:
        torch.Tensor: ガウシアンフィルタを適用した結果。
    """
    original_dim = x.dim()
    if original_dim == 3:
        x_batched = x.unsqueeze(0) # (C, H, W)の場合、バッチ次元を追加して (1, C, H, W) にする
    elif original_dim == 4:
        x_batched = x
    else:
        raise ValueError("Unsupported tensor dimensions. Expected 3 or 4 dimensions.")

    # フィルタのパラメータ設定 (元の関数から流用)
    ksize = 7
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    channels = x_batched.shape[1]
    
    # カーネルのキャッシュキーを生成
    ksize_tuple = (ksize, ksize)
    dtype_str = str(dtype)

    # キャッシュをチェック
    if ksize_tuple in _gauss_ker_cache[sigma] and dtype_str in _gauss_ker_cache[sigma][ksize_tuple]:
        kernel_2d = _gauss_ker_cache[sigma][ksize_tuple][dtype_str].to(device, dtype=dtype)
    else:
        # カーネルの生成（1D）
        kernel_1d = cv2.getGaussianKernel(ksize, sigma)
        kernel_1d = torch.from_numpy(kernel_1d).to(dtype).to(device)
        
        # 2Dカーネルに変換 (1, 1, H, W)
        kernel_2d = torch.matmul(kernel_1d, kernel_1d.T).unsqueeze(0).unsqueeze(0)

        # キャッシュに格納 (チャネル軸を持つ前の 2D カーネルを CPU に格納)
        if ksize_tuple not in _gauss_ker_cache[sigma]:
            _gauss_ker_cache[sigma][ksize_tuple] = {}
            
        _gauss_ker_cache[sigma][ksize_tuple][dtype_str] = kernel_2d.detach().clone().cpu()

    # チャネル数に対応させる (C, 1, H, W)
    kernel = kernel_2d.repeat(channels, 1, 1, 1)
    
    # ガウシアンフィルタを適用
    # `groups=channels`でチャネルごとの畳み込み、`padding`は ksize から自動計算
    padding = ksize // 2 
    
    filtered_x = torch.nn.functional.conv2d(
        input=x_batched.float(),
        weight=kernel.float(),
        padding=padding, 
        groups=channels
    )
    filtered_x = filtered_x.to(dtype)

    # 元の次元数に戻す
    if original_dim == 3:
        filtered_x = filtered_x.squeeze(0) # (1, C, H, W) -> (C, H, W)
    
    return filtered_x

# ==============================================================================================


def tensor_to_edges(x, is_dist_detect, dtype, device):
    # --- Canny法 + 距離変換によるエッジ検出（距離マップの生成） ---
    # x: input_tensor。　例：noise_pred,target

    # NaN/Infを丸める (loss計算前の最終安全策)
    x = torch.nan_to_num(x, nan=0.0, posinf=1e-4, neginf=0.0)    
        
    # 距離変換の設定
    MAX_DIST_CLIP = 80.0  # 距離の上限 (ラテント空間での約10px相当)
    GAMMA_DIST = 0.7      # 距離の滑らかさ調整

    x = x.detach().to(torch.float32).cpu().numpy()
    edges_all = []
    
    for i in range(x.shape[0]):
        img = np.transpose(x[i], (1, 2, 0)) # (C,H,W)→(H,W,C)
        
        # 無効な値 (NaN, Inf) が含まれる場所を記録
        invalid_mask = ~np.isfinite(img).all(axis=2)
        
        # 不正値処理とクリッピングを追加
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        img = np.clip(img, 0.0, 1.0)
        
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # 1. Canny法でエッジを検出 (255=エッジ, 0=背景)
        edges = cv2.Canny(
            gray, 
            200, 
            500,    # latentが対象なので、厳しくなければノイズを拾ってしまう
            L2gradient=True
        )
        
        if is_dist_detect:
            # 2. 距離変換のための反転 (0=エッジ, 255=背景)
            edges_inv = 255 - edges
            
            # 3. 距離変換の実行 (エッジ=0.0, 離れるほど値が増加)
            distance_map = cv2.distanceTransform(
                edges_inv, 
                cv2.DIST_L2, 
                cv2.DIST_MASK_PRECISE
            )
            
            # 4. クリッピングと正規化 （背景は255 → 1.0に変化）
            distance_map = np.clip(distance_map, 0.0, MAX_DIST_CLIP)
            normalized_dist = distance_map / MAX_DIST_CLIP
            
            # 5. 滑らかさ調整 (ガンマ補正)
            edges = np.power(normalized_dist, GAMMA_DIST)

        else:
            edges = edges.astype(np.float32) / 255.0
 
        # 無効領域の値を、そのモードの背景色に合わせる
        if is_dist_detect:
            edges[invalid_mask] = 1.0  # 距離変換時は 1.0 (最大距離)
        else:
            edges[invalid_mask] = 0.0  # 通常時は 0.0 (エッジなし)
        
        edges_all.append(edges)
        
    edges_all = np.stack(edges_all, axis=0)
    edges_tensor = torch.from_numpy(edges_all).unsqueeze(1).to(device, dtype=dtype)
    return edges_tensor
        
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
            
    return loss, pred_norm
    
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
        return torch.zeros(1, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype)

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
    
    return loss, pred_flows_flat

def calc_loss_pool(target, noise_pred, args, huber_c, reso_scale, is_above_limit):
    # 画像単体のpool分割したうえで、それぞれの領域を比較する。
    # これがあることで、画像単体としてのバランスや、人物の基本骨格が取れるようになる
    # 骨格が一致しなければ、あらゆる詳細学習が進まない
    
    dtype = target.dtype
    device = target.device    
    
    if not is_above_limit:
        # 解像度が低い場合、信頼性が著しく低下する・・・気がしたけど、いい感じに分割してくれるので、is_above_limit=True運用で問題ない
        return torch.zeros(1, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype)

    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)

    def extract_features(x):
        # 空間情報の抽出：統計量を測定し、特徴を際立たせる
        
        # (B, C, H, W) -> (B, C, H*W) : チャンネルごとの画素平坦化
        x_flat = x.flatten(2)
        device = x.device

        features = []    
    
        pool_1x  = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        pool_3x  = torch.nn.functional.adaptive_avg_pool2d(x, (3, 3))
        pool_5x  = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))        
                             
        features = [
            pool_1x.flatten(1), 
            pool_3x.flatten(1), 
            pool_5x.flatten(1), 
        ]
        
        return torch.cat(features, dim=1)
    
    # 特徴抽出と標準化を一括処理
    pool_pred   = extract_features(target_latents)
    pool_target = extract_features(pred_latents)
    
    boost=1.0 
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

    return loss, pool_pred

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
    
    if not is_above_limit:
        return torch.zeros(1, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype)

    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)

    def apply_distance_weight(i, j, num_grid_w, dtype, device):
        """
        グリッド上の距離に応じて重みを生成
        1マス隣: 3.0倍 / 2マス隣: 2.0倍 / その他: 1.0倍
        """
        # インデックスから2次元座標(x, y)を復元
        dx = (i % num_grid_w - j % num_grid_w).abs().to(dtype)
        dy = (i // num_grid_w - j // num_grid_w).abs().to(dtype)
        
        # 直線距離(ユークリッド距離)に基づいて算出
        dist = torch.sqrt(dx**2 + dy**2)

        # 1.0(ベース) + 強度 * 指数減衰
        # dist=1.0 のとき 3.0 になるよう設定
        strength = 2.0
        decay = 0.8 
        weights = 1.0 + strength * torch.exp(-decay * (dist - 1.0))
        
        return weights

    # サイズから分割数決定
    H, W, _ = get_image_hw(target)
    scale_latents = scale_px / 8 # px → latents
    num_grid_h = int(max(1, H // scale_latents))
    num_grid_w = int(max(1, W // scale_latents))
    #print(f"scale_latents:\t{scale_latents}\tnum_grid_h/w:\t{num_grid_h}\t{num_grid_w}")

    # 1. 空間情報の抽出
    target_small = torch.nn.functional.adaptive_avg_pool2d(target_latents.float(), (num_grid_h, num_grid_w))
    pred_small = torch.nn.functional.adaptive_avg_pool2d(pred_latents.float(), (num_grid_h, num_grid_w))
    
    # 2. 特徴ベクトル化 (B, HW, C)
    target_feat = target_small.flatten(2).transpose(1, 2)
    pred_feat = pred_small.flatten(2).transpose(1, 2)
    
    # 3. 自己相関行列 (B, HW, HW) の生成
    target_corr = torch.bmm(target_feat, target_feat.transpose(1, 2))
    pred_corr = torch.bmm(pred_feat, pred_feat.transpose(1, 2))
    
    # 1点あたりの相関強度に落とし込む → 必要なしと判断。どれも重要な変化だし、値が小さくなりすぎる
    num_locations = num_grid_h * num_grid_w
    #target_corr = target_corr / num_locations
    #pred_corr = pred_corr / num_locations

    # 重複と自己相関を排除するインデックスを抽出。 (B, HW*HW) -> (B, 無効ペアを除外)
    # torch.triu_indices(row, col, offset=1) により、対角成分(0)を除いた上三角のみ取得
    indices = torch.triu_indices(num_locations, num_locations, offset=1, device=device)
    i, j = indices[0], indices[1]  # (i:基準点, j:比較点) 
    target_pairs = target_corr[:, i, j]
    pred_pairs   = pred_corr[:, i, j]
    
    # 距離重みの取得と適用
    dist_weight = apply_distance_weight(i, j, num_grid_w, dtype, device)
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
        loss_type=args.loss_type, 
        huber_c=huber_c
    )
            
    return loss, pred_pairs
    
def calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit, mode):
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
    

    if not is_above_limit:
        # 解像度が低い場合、いくつかの統計値に対する信頼性が著しく低下する
        return torch.zeros(1, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype)
        
    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_latents = target if is_batched else target.unsqueeze(0)
    pred_latents = noise_pred if is_batched else noise_pred.unsqueeze(0)          
        
    batch_size = target_latents.shape[0]        
    if batch_size < 2:
        # batch_size = 1なら計算する目的がない
        return torch.zeros(1, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype)
        
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
            
            pool_1x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            pool_3x = torch.nn.functional.adaptive_avg_pool2d(x, (3, 3))
            pool_5x = torch.nn.functional.adaptive_avg_pool2d(x, (5, 5))
            d3_1 = pool_3x - pool_1x
            d5_3 = pool_5x - torch.nn.functional.interpolate(pool_3x, size=(5, 5), mode='bilinear')
            
                                 
            features = [
                pool_1x.flatten(1), 
                (d3_1.flatten(1) * 3.0) / ( 3 ** 2),  # = 個別ブースト3+2合計で５倍。要素数で割って信号を平均化、
                (d5_3.flatten(1) * 2.0) / ( 5 ** 2),
            ]
            
            num_features    = 2     # pool_1xとそれ以外の２種類。細かい内訳は、features内で計算
            boost           = 1.0   # 体感上このくらいがbaseと同程度になる

        elif mode == "pool_tones": # まだ使用不可。研究中。そもそもなくても十分な気もするし、リスキー
            sampling_reso = 20
            q_list = torch.linspace(0.0, 1.0, sampling_reso, device=device, dtype=torch.float32)
            b, c, h, w = x.shape

            # 1x1 (全体) の tones
            x_delta_all = x_flat - x_flat.mean(dim=2, keepdim=True)
            tones_1x = torch.quantile(x_delta_all.float(), q_list, dim=2).permute(1, 2, 0).flatten(1)

            # 3x3 (局所) の tones
            grid = 3
            p = x.unfold(2, h//grid, h//grid).unfold(3, w//grid, w//grid)
            p = p.contiguous().view(b, c, grid, grid, -1).permute(0, 2, 3, 1, 4)
            p_delta = p - p.mean(dim=-1, keepdim=True)
            tones_3x = torch.quantile(p_delta.float(), q_list, dim=-1).permute(1, 2, 3, 4, 0).flatten(1)

            features = [tones_1x, tones_3x]
            num_features = sampling_reso * (1 + grid * grid)
            boost = 1.5
            
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
    # 類似度という意味では内積方式がいいのだが、L2特性になってしまい、値が過小傾向になってハズレ値ばかり優先されてしまう。
    # meanではなくsumを使用することで、各誤差の相殺・対消滅を防ぐ
    rel_pred    = (feat_pred.unsqueeze(1)   - feat_pred.unsqueeze(0)).abs().sum(dim=2)
    rel_target  = (feat_target.unsqueeze(1) - feat_target.unsqueeze(0)).abs().sum(dim=2)
    
    eps = 1e-8 # アンダーフロー対策
    rel_pred    = (rel_pred + eps) / num_features
    rel_target  = (rel_target + eps) / num_features
    
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
    
    return loss, rel_pred
    
#-----------------------------------------

_LOSS_CONFIG = {
    #名称と、(重み倍率, gamma, deadband)
    # gamma     ：lossを何乗するか。大きいほど学習後期よりも学習序盤に寄与する。
    # deadband  ：この閾値以下のlossをカットして、過適合を防ぐ
    "base   ":  (1.0, 1.0, 0.0),    # 最も大切なlossではあるが、grad/loss効率が低いので、強調したいところ
    "pool    ": (1.0, 1.0, 0.0),
    "ch_cosine": (0.5, 1.0, 0.1),
    "ch_flow":  (0.5, 1.0, 0.1),
    "pair_128px": (0.5, 1.0, 0.0),
    "pair_64px": (0.5, 1.0, 0.0),
    "batch_pool": (0.5, 1.0, 0.1), 
    "batch_cos": (0.5, 1.0, 0.1), 
}

_LOSS_NAMES = list(_LOSS_CONFIG.keys())

# 各ロスの過去20ステップの平均値をfloatで記録する辞書
_LOSS_HISTORY = {name: collections.deque(maxlen=20) for name in _LOSS_CONFIG.keys()}
    
def combine_losses_dynamically(
    losses_list: list[torch.Tensor], 
    global_step,
) -> torch.Tensor:
    if not losses_list:
        return torch.tensor(0.0, device='cpu')

    loss_base_current = losses_list[0]       
    
    first_valid_loss = next(l for l in losses_list if l is not None)
    first_valid_loss_tensor = first_valid_loss[0] if isinstance(first_valid_loss, tuple) else first_valid_loss
    all_loss = torch.zeros_like(first_valid_loss_tensor)
        
    num_targetable_losses = 0
    for i, loss_value_raw in enumerate(losses_list):
        if i < len(_LOSS_NAMES):
            if loss_value_raw is not None:
                num_targetable_losses += 1
    
    grads_list = []
    computed_losses_scalar = []
        
    # 基準となる形状（通常は最初の有効なロス）を特定するための準備
    base_shape_tensor = None
    
    # PCgradで合成できない、たまに発生するスカラー
    scalar_only_sum = torch.tensor(0.0, device=first_valid_loss_tensor.device)
    
    # 1. 各損失の勾配算出とリスト化
    for i, item in enumerate(losses_list):
        if i < len(_LOSS_NAMES) and item is not None:
            loss_value_raw, pred = item

            loss_name = _LOSS_NAMES[i]
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
            
            if global_step % 50 == 0 or is_debug_mode:
                def loss_bar(loss):
                    max_bar = 10  # 0.05刻みで最大0.5 → 10段階
                    capped_loss = min(loss, 0.5)
                    blocks = int(capped_loss / 0.05)
                    bar = "█" * blocks + " " * (max_bar - blocks)
                    return bar
                bar = loss_bar(loss_instance.mean().item())
                
                indent = "\n" if i == 0 else ""
                print(f"{indent} {loss_name} \tgamma:{base_gamma}*{gamma_value}\tSt_wt:{static_weight:.3f} \tDy_wt:{dynamic_weight:.3f} \tloss補正前/補正後\t{current_loss_mean.item():.3f}/{loss_instance.mean().item():.3f}\t|{bar}|")
            
            
            # PCgradの処理---------------------------------------------
            
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
            else:
                # グラフがつながっていない場合は、形状を合わせたゼロ勾配を代入
                grad = torch.zeros_like(pred).detach()
                
            grads_list.append(grad)
            computed_losses_scalar.append(loss_scalar)
            
            if base_shape_tensor is None:
                base_shape_tensor = loss_value_raw

    if not grads_list:
        return torch.tensor(0.0, device=losses_list[0].device, requires_grad=True)

    # 2. PCGrad：全勾配間の衝突を並列に解消
    num_losses = len(grads_list)
    
    # 【最適化】二重ループ外で形状情報を整理し、最大要素数でパディングを適用
    max_numel = max(g.numel() for g in grads_list)
    flat_grads = []
    original_shapes = []
    for g in grads_list:
        original_shapes.append(g.shape)
        g_f = g.reshape(-1)
        if g_f.numel() < max_numel:
            g_f = torch.nn.functional.pad(g_f, (0, max_numel - g_f.numel()))
        flat_grads.append(g_f)

    edited_flat_grads = [g.clone() for g in flat_grads]
    indices = list(range(num_losses))
    random.shuffle(indices)

    # デバッグ専用統計値
    conflict_count = 0
    total_reduction_norm = 0.0
    original_norms_sum = 0.0

    for i in indices:
        gi_flat = edited_flat_grads[i]
        
        for j in indices:
            if i == j: continue
            
            gj_flat = flat_grads[j] # 比較対象は事前平坦化済みテンソル
            
            if gi_flat.dtype != gj_flat.dtype:
                gj_flat = gj_flat.to(gi_flat.dtype)
                
            # 計算前に nan/inf を 0 に置換（安全装置）
            gi_flat = torch.nan_to_num(gi_flat, nan=0.0, posinf=0.0, neginf=0.0)
            gj_flat = torch.nan_to_num(gj_flat, nan=0.0, posinf=0.0, neginf=0.0)

            dot_prod = torch.dot(gi_flat, gj_flat)

            if is_debug_mode_PCgrad:
                before_norm = torch.norm(gi_flat) # 修正前のノルムを記録
            
            if dot_prod < 0:
                # 異方向（干渉対策）
                # 直交成分だけを残し、干渉成分を相殺する
                conflict_count += 1
                                
                mag_sq = torch.dot(gj_flat, gj_flat) + 1e-8
                gi_flat = gi_flat - (dot_prod / mag_sq) * gj_flat

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
        edited_flat_grads[i] = gi_flat

    # 【最適化】平坦化したテンソルを元の形状に復元して edited_grads を作成
    edited_grads = []
    for k, efg in enumerate(edited_flat_grads):
        orig_s = original_shapes[k]
        actual_numel = torch.prod(torch.tensor(orig_s)).item()
        edited_grads.append(efg[:actual_numel].view(orig_s))

    # 3. 内部関数：形状の不一致を補正して加算
    def _accumulate_with_shape_match(target_tensor, source_tensor):
        if source_tensor is None or source_tensor.numel() == 0:
            return target_tensor

        # スカラー(dim=0)や1次元(dim=1)の勾配を弾く
        if source_tensor.dim() < 2:
            return target_tensor
        
        eg = source_tensor.clone()
        if eg.shape != target_tensor.shape:
            # 2Dテンソル (B, C) を (B, C, 1, 1) に展開
            if eg.dim() == 2:
                eg = eg.unsqueeze(-1).unsqueeze(-1)
            
            # チャンネル数（次元1）の調整
            if eg.shape[1] != target_tensor.shape[1] and eg.shape[1] > 0:
                if target_tensor.shape[1] % eg.shape[1] == 0:
                    repeat_factor = target_tensor.shape[1] // eg.shape[1]
                    eg = eg.repeat(1, repeat_factor, 1, 1)
                else:
                    return target_tensor # 合成不能な場合はそのまま返す

            # 空間解像度 (H, W) のリサイズ
            if eg.shape[2:] != target_tensor.shape[2:]:
                if any(s == 0 for s in eg.shape):
                    return target_tensor
                
                eg = torch.nn.functional.interpolate(
                    eg, 
                    size=target_tensor.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # 形状一致を確認して加算
        if eg.shape == target_tensor.shape:
            target_tensor += eg
            
        return target_tensor

    accumulated_grad = torch.zeros_like(grads_list[0])
    for eg in edited_grads:
        accumulated_grad = _accumulate_with_shape_match(accumulated_grad, eg)
    
    # 4. 最終的なテンソルとしての再構成
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
            
            # 各ロスの重みとガンマを適用
            static_weight, gamma_value, _ = _LOSS_CONFIG.get(_LOSS_NAMES[i], (1.0, 1.0, 0.0))
            total_loss_tensor += (l_val ** gamma_value) * static_weight

    # 勾配すり替え：値は total_loss_tensor、勾配は PCGrad後の accumulated_grad
    # これにより、呼び出し側で .mean([1, 2, 3]) をしても正しく勾配が伝播します
    # accumulated_grad - accumulated_gradは、値は0だが、勾配の計算グラフ（履歴）だけを繋ぐ
    all_loss = total_loss_tensor + (accumulated_grad - accumulated_grad).detach()
    all_loss += scalar_only_sum

    if is_debug_mode_PCgrad:     
        # PCGrad 統計値の計算
        reduction_rate = (total_reduction_norm / (original_norms_sum + 1e-8)) * 100
        print(f" [PCGrad Stats] Conflicts(+-unmatch): {conflict_count} | Grad Cut Rate: {reduction_rate:.2f}%")
            

        """
        if is_debug_mode:
            is_trainable = loss_value.grad_fn is not None
            is_trainable_all = all_loss.grad_fn is not None            
            grad_indicator = "● (Grad OK)" if is_trainable else "× (No Grad/Static)"
            grad_indicator_all = "● (Grad OK)" if is_trainable_all else "× (No Grad/Static)"
            print(f"勾配追跡可能か：\tloss単体\累計後\t{grad_indicator}\t{grad_indicator_all}")
        """
                
                   
    return all_loss

# モジュールレベルにロス履歴を保存するdequeを追加
_LOSS_BASE_HISTORY = collections.deque(maxlen=20)

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

    loss_pool, pred_pool = calc_loss_pool(target, noise_pred, args, huber_c, reso_scale, is_above_limit=True)
    loss_ch_cosine, pred_ch_cosine = calc_loss_ch_cosine(target, noise_pred, args, huber_c, reso_scale)
    loss_ch_flow, pred_ch_flow = calc_loss_ch_flow_2(target, noise_pred, args, huber_c, reso_scale, is_above_limit)
    loss_pair_corr_128px, pred_pair_128px = calc_loss_pair_correlation(target, noise_pred, args, huber_c, is_above_limit, scale_px=128)
    loss_pair_corr_64px, pred_pair_64px = calc_loss_pair_correlation(target, noise_pred, args, huber_c, is_above_limit, scale_px=64)
    loss_batch_pool, pred_batch_pool = calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit, mode="pool")
    loss_batch_cos, pred_batch_cos = calc_loss_batch_relation(target, noise_pred, args, huber_c, reso_scale, is_above_limit=True, mode="ch_cosine")
   
    # 統合するlossをリスト化する。
    # リストの位置が重要なので、必ず何かを代入すること。統合をスキップしたい場合はNoneを代入する。
    all_computed_losses = [
        (loss_base, noise_pred),
        (loss_pool, pred_pool),
        (loss_ch_cosine, pred_ch_cosine),
        (loss_ch_flow * _current_snr_weight, pred_ch_flow),
        (loss_pair_corr_128px, pred_pair_128px),
        (loss_pair_corr_64px, pred_pair_64px),
        (loss_batch_pool, pred_batch_pool),
        (loss_batch_cos, pred_batch_cos),
    ]
    
    # NaN/Inf補正    
    all_computed_losses = [
        (torch.nan_to_num(l, nan=0.0, posinf=1e-4, neginf=0.0), p) if l is not None else None 
        for l, p in all_computed_losses
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
    
    """
    # デバッグ（重み補正前）
    if global_step % 50 == 0:
        all_loss_values = [f"{l.mean().item():.4f}" for l in all_computed_losses if l is not None]
        accelerator.print(f"loss内訳: {', '.join(all_loss_values)}")
    """
        
    # ロスの集計と重み付け
    loss = combine_losses_dynamically(all_computed_losses, global_step)

    # VRAM解放のため、個々の損失テンソルを削除
    for l in all_computed_losses:
        if l is not None:
            del l
    
    # 不要な変数を削除
    del all_computed_losses
    
    return loss
