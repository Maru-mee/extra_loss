# -------------------------------------------
# loss_extra_calc
# 画像生成AIにおいて、様々なlossの統計結果を生成します
# -------------------------------------------

# 開発者向けオプション -----------------------------------------
is_print_screen         = True  # Falseにより、一切の画面表示をOFF
print_interval_step     = 50    # print表示のstep間隔。

is_debug_mode           = False
is_debug_mode_grad      = False
is_debug_mode_PCgrad    = False
is_logging_grad         = False
# ---------------------------------------------------------

import collections
import cv2
import math
import random
import torch
import torch.nn.functional
from torchvision.transforms import RandomCrop

import library.train_util as train_util
from library.custom_train_functions import (
    apply_masked_loss,
)

if is_logging_grad:        
    try:
        from .debug import grad_logger
    except (ImportError, ValueError):
        try:
            import debug.grad_logger as grad_logger
        except ImportError:
            print("warning: grad_logger not found in debug folder")        

_current_snr_weight = None
_current_mask = None
_random_seed_1 = 0
_dtype = None
_device = None

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
    area_img = area_latents * 64
    return H, W, area_latents, area_img

# キャッシュを格納する辞書
_gauss_ker_cache = collections.defaultdict(dict)
def filtering_gaussian(x):
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

    # フィルタのパラメータ設定 (元の関数から流用)
    ksize = 3
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    channels = x.shape[1]
    
    # カーネルのキャッシュキーを生成
    ksize_tuple = (ksize, ksize)
    dtype_str = str(_dtype)

    # キャッシュをチェック
    if ksize_tuple in _gauss_ker_cache[sigma] and dtype_str in _gauss_ker_cache[sigma][ksize_tuple]:
        kernel_2d = _gauss_ker_cache[sigma][ksize_tuple][dtype_str].to(_device, dtype=_dtype)
    else:
        # カーネルの生成（1D）
        kernel_1d = cv2.getGaussianKernel(ksize, sigma)
        kernel_1d = torch.from_numpy(kernel_1d).to(_dtype).to(_device)
        
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
        input=x.float(),
        weight=kernel.float(),
        padding=padding, 
        groups=channels
    )
    filtered_x = filtered_x.to(_dtype)
    
    return filtered_x

def adaptive_avg_pool2d_for_latents(input, output_size):
    """
    latentsに最適化したadaptive_avg_pool2d
    adaptive_avg_pool2dは平均しか評価せず、latentsの大きい変化を評価できない。
    ｓｔdを追加することで、latentsの統計値を評価できるようにする。
    
    引数の定義：
        torch.nn.functional.adaptive_avg_pool2dと同じ
    """
    eps = 1e-10

    # inputとその2乗をチャネル方向に結合して、プール計算を一回にまとめる
    combined = torch.cat([input, input.pow(2)], dim=1)
    pooled = torch.nn.functional.adaptive_avg_pool2d(combined, output_size)

    mean, mean_sq = torch.chunk(pooled, 2, dim=1)
    std = torch.sqrt(mean_sq - mean.pow(2) + eps)
    
    return mean + std

def get_ch_vector(x):
    # latents情報をベクトル表現に直して、勾配予測をアシストする
    # xが、target, noise_predのときは、各ピクセルにおけるチャンネル方向の成分比のベクトルを返す

    eps = 1e-10    
    direction = torch.nn.functional.normalize(x.float(), p=2, dim=1, eps=eps)
    magnitude = torch.abs(x) # 厳密にはsqrt(x^2)とするべきだが計算コストが増加するだけだし、grad導出時の導関数が=1/√(x^2)となり収束期にゼロ除算リスクを生む
    vector = (direction * magnitude).to(_dtype)
    return vector
    
def get_batch_vector(x):
    # バッチ方向のベクトル化

    eps = 1e-10     
    direction = torch.nn.functional.normalize(x.float(), p=2, dim=0, eps=eps)
    magnitude = torch.abs(x)
    vector = (direction * magnitude).to(_dtype)
    return vector
    
def get_pair_vector(x):
    # loss_pair専用のベクトル化
    # x: (B, N_pairs, C)
    
    eps = 1e-10
    direction = torch.nn.functional.normalize(x.float(), p=2, dim=2, eps=eps)
    magnitude = torch.abs(x)
    vector = (direction * magnitude).to(_dtype)
    return vector
    
# ==============================================================================================

def calc_loss_pool(target, noise_pred, args, huber_c, is_above_limit, pool_num):
    """
    画像単体のpool分割したうえで、それぞれの領域を比較する。
    これがあることで、画像単体としてのバランスや、人物の基本骨格が取れるようになる
    骨格が一致しなければ、あらゆる詳細学習が進まない
    しかし、latentsにおけるmean比較というのは茶色くなりがちなので、強い強度での適用は控えたほうがいい
    """
    
    if not pool_num == 1: # pool_num=1を特別に許可する。平均色の学習としてはどんな解像度でも有意義
        if not is_above_limit:  # 解像度が低い場合、信頼性が著しく低下する
            return torch.zeros(1, device=_device, dtype=_dtype)

    def extract_features(x, pool_num):
        # 空間情報の抽出：統計量を測定し、特徴を際立たせる
                    
        pool_x  = adaptive_avg_pool2d_for_latents(x.float(), (pool_num, pool_num))
                             
        features = [
            pool_x.flatten(1), 
        ]
        
        return torch.cat(features, dim=1)
    
    # 特徴抽出と標準化を一括処理
    feat_pred   = extract_features(target, pool_num)
    feat_target = extract_features(noise_pred, pool_num)
    
    boost = 0.1
    scales = boost
    
    feat_pred.mul_(scales)
    feat_target.mul_(scales)
    
    loss = train_util.conditional_loss(
        feat_pred.float(),
        feat_target.float(),
        reduction="none",
        loss_type=args.loss_type,
        huber_c=huber_c
    )
    
    return loss
    
def calc_loss_ch_vector(target, noise_pred, args, huber_c):
    """
    ピクセル単位のチャネル間ベクトルの一致度を算出。
    色相や概念の向きを評価
    pixel perfectやドット検出（多少ではあるがエッジ検出）の効果があり、新しい画像要素を発見するのに特に有効。
    どのtimestepでもそこそこの効果を持つ、MSE同等以上の使い勝手を持つ
    """

    feat_target   = get_ch_vector(target)
    feat_pred     = get_ch_vector(noise_pred)
    
    loss = train_util.conditional_loss(
        feat_pred.float(),
        feat_target.float(),
        reduction="none",
        loss_type=args.loss_type,
        huber_c=huber_c
    )
            
    return loss


def calc_loss_ch_flow_2(target, noise_pred, args, huber_c, is_above_limit, searching_radius=2.0):
    """
    連続座標サンプリングによるベクトル相関を全方位・等距離で同期。
    真円状のエッジ検出に優れている。ピクセル単位ではなく、該当距離（ピクセルの隙間含む）の値を滑らかに取るためノイズに強い
    loss_ch_vectorで低下するピクセル間の連続性（ケロイドなどの学習の副産物）を抑制する
    
    searching_radius :検索半径[px] 相当距離であって、pxそのものではない。単一の値か、リスト（複数の半径）かを指定
    """

    eps = 1e-10

    if not is_above_limit:
        # 解像度が低い場合、計算困難な境界影響が強くなり、境界クロップが発生しやすくなる
        return torch.zeros(1, device=_device, dtype=_dtype)
        
    if not isinstance(searching_radius, list):
        searching_radius = [searching_radius]  

    def create_base_grid(B, H, W):
        # 基準となる座標の網を作る
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=_device, dtype=_dtype),
            torch.linspace(-1, 1, W, device=_device, dtype=_dtype),
            indexing='ij'
        )
        return torch.stack([grid_x, grid_y], dim=-1).to(_dtype).unsqueeze(0).expand(B, -1, -1, -1)

    def sample_by_angle(latents, base_grid, angle, r, step_h, step_w):
        # 指定した角度に網をずらして、座標を吸い出す
        
        offset_x = math.cos(angle) * r * step_w
        offset_y = math.sin(angle) * r * step_h
        
        sampling_grid = base_grid.to(_dtype).clone()
        sampling_grid[..., 0] += offset_x
        sampling_grid[..., 1] += offset_y
        
        return sampling_grid

    def calc_vector_diff(orig_latents, sampled_latents, mask):
        # 中心点origに対する、sample点情報を計算
        
        #ターゲットの差分を重みとする。
        valid_indices = mask.unsqueeze(1).expand_as(orig_latents)
        
        # 中心点とサンプリング点をベクトル化
        diff = orig_latents - sampled_latents
        
        vector = get_ch_vector(diff)
        
        return vector[valid_indices]

    def get_ch_flow(target, pred):
        H, W, _, _ = get_image_hw(target)
        B, C, _, _ = target.shape
        base_grid = create_base_grid(B, H, W)
        
        step_h, step_w = 2.0 / (H - 1), 2.0 / (W - 1)
        angles = torch.linspace(0, 2 * math.pi, 9, device=_device)[:-1]
        
        target_list, pred_list = [], []
        
        for r in searching_radius:
            for angle in angles:

                sampling_grid = sample_by_angle(target, base_grid, angle.item(), r, step_h, step_w)
                mask = (sampling_grid[..., 0].abs() <= 1) & (sampling_grid[..., 1].abs() <= 1) # 有効領域のみ抽出する（画像の外を対象外とする）
                
                # grid付近のピクセルから、値を補間しつつ取得する
                # 補足：padding_mode='zeros'の場合、画像端にある被写体が「黒い壁」と隣接していると判定され、そこに実在しない強烈なコントラスト（偽のエッジ）が発生するリスク有り
                sampled_target = torch.nn.functional.grid_sample(
                    target.float(), sampling_grid.float(), mode='bilinear', padding_mode='border', align_corners=True
                )
                sampled_pred = torch.nn.functional.grid_sample(
                    pred.float(), sampling_grid.float(), mode='bilinear', padding_mode='border', align_corners=True
                )
                
                # 基準と比較対象との差分を計算
                target_list.append(calc_vector_diff(target, sampled_target, mask))
                pred_list.append(calc_vector_diff(pred, sampled_pred, mask))
            
        return torch.cat(target_list), torch.cat(pred_list)

    feat_target, feat_pred = get_ch_flow(target, noise_pred)
    
    loss = train_util.conditional_loss(
        feat_pred.float(), 
        feat_target.float(),
        reduction="none", 
        loss_type=args.loss_type, 
        huber_c=huber_c
    )
    
    return loss

def calc_loss_sparsity(target, noise_pred, args, huber_c):
    """
    チャンネル間の情報の尖り具合（スパース性）を同期させる。
    一般的なloss_MSEは、茶色やグレー単色の画像を好む。なぜならば、それが最も手軽に到達できる平均解であるため。
    このlossは、必要な尖り状態があるかを評価することで、安易な平均色化へペナルティをかける
    光源などの発色を得るために効果的
    
    メインの対象物だけでなく、サブの対象も学習しようとする力が働く。
    
    使用上の注意：ノイズ成分がL2ノルムを不当に底上げするため計算結果が不安定になりがち。
    target,noise_predにはガウシアンフィルタを予め適用しておくか、不安定にならない数式を使用すること
    """
    eps = 1e-10

    l1_pred = torch.abs(noise_pred).sum(dim=1)
    l1_target = torch.abs(target).sum(dim=1)
    
    l2_pred = torch.sqrt(torch.pow(noise_pred, 2).sum(dim=1) + eps)
    l2_target = torch.sqrt(torch.pow(target, 2).sum(dim=1) + eps) 
    
    # ベース設計。L2の値次第で不安定になる可能性があるので廃止
    # feat_pred = l1_pred / l2_pred
    # feat_target = l1_target / l2_target
    
    l1_pred = torch.clamp(l1_pred, min=eps)
    l2_pred = torch.clamp(l2_pred, min=eps)
    l1_target = torch.clamp(l1_target, min=eps)
    l2_target = torch.clamp(l2_target, min=eps)    
    
    feat_pred   = torch.log(l1_pred)    - torch.log(l2_pred)
    feat_target = torch.log(l1_target)  - torch.log(l2_target)

    loss = train_util.conditional_loss(
        feat_pred.float(),
        feat_target.float(),
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
    
    # サイズから分割数決定
    H, W, _, _ = get_image_hw(target)
    scale_latents = scale_px / 8 # px → latents
    num_grid_h = int(max(1, H // scale_latents))
    num_grid_w = int(max(1, W // scale_latents))
    
    # 4x4を確保できないならば精度不足。120通りのペア数を保証することで多様性を確保
    if num_grid_h < 4 or num_grid_w < 4:
        return torch.zeros(1, device=_device, dtype=_dtype)
        
    # ペア番地の作成 -------------------------------------------    
    # ペア間の距離制限を設けることだけが目的。距離制限を考慮しないのならば、不要な行程

    dist_max = 5 # 単位: grid。基準となるグリッドに対して、半径方向何gridまでの正方形領域をペア対象とするか

    grid_y, grid_x = torch.meshgrid(
        torch.arange(num_grid_h, device=_device), 
        torch.arange(num_grid_w, device=_device), 
        indexing='ij'
    )
    coords = torch.stack(
        [grid_y.flatten(), grid_x.flatten()], 
        dim=1
    )

    num_locations = num_grid_h * num_grid_w
    indices = torch.triu_indices(num_locations, num_locations, offset=1, device=_device)
    i, j = indices[0], indices[1]
    dist = (coords[i] - coords[j]).abs().sum(-1)
    i, j = i[dist <= dist_max], j[dist <= dist_max]
        
    # -------------------------------------------------------]
    
    # 空間情報の抽出
    feat_target = adaptive_avg_pool2d_for_latents(target.float(), (num_grid_h, num_grid_w))
    feat_pred   = adaptive_avg_pool2d_for_latents(noise_pred.float(), (num_grid_h, num_grid_w))
    
    # 特徴ベクトル化 (B, HW, C)
    feat_target = feat_target.flatten(2).transpose(1, 2)
    feat_pred = feat_pred.flatten(2).transpose(1, 2)
    
    # 相関行列の生成（要素ごとの差分で番地情報を維持）
    # (B, HW, 1, C) - (B, 1, HW, C) -> (B, HW, HW, C)
    feat_target = feat_target[:, i, :] - feat_target[:, j, :]
    feat_pred = feat_pred[:, i, :] - feat_pred[:, j, :]
    
    # ベクトル化
    # これによって、loss_type=l1時の、grad_absのmaxとmeanが等しくなってしまう問題を解消
    feat_target = get_pair_vector(feat_target)
    feat_pred   = get_pair_vector(feat_pred)
    
    loss = train_util.conditional_loss(
        feat_pred, 
        feat_target,
        reduction="none", 
        loss_type=args.loss_type,
        huber_c=huber_c
    )
            
    return loss
    
def calc_loss_batch_relation(
    target, noise_pred, args, huber_c, area_latents, is_above_limit, mode, pool_num=1,
):
    """
    ・バッチ内の画像間の類似度構造（Relation）をターゲットと同期させることで、
      各画像が持つ固有の特徴的な差異を学習し、トークンの定着率を高める。
    ・特定の画像の学習に偏った学習をさせない。メジャーなプロンプトへ収束してしまう現象を抑制
    ・UNet学習済みの画像パターンA,パターンBに無理やり当てはめるような「手抜き生成」を防ぎ、
      入力トークンごとの微細な描き分けをモデルに強制させる。
    ・テンプレ的な「AIらしさ」や「平均的な綺麗さ」は減るが、プロンプトへの忠実度は向上する。
    ・snrが異なると正しい比較ができない。snrごとの比較を行う。
    """
    
    batch_size = target.shape[0]
    
    total_loss_relation = torch.zeros(batch_size, device=_device, dtype=_dtype)
    is_execute_flag = True
    
    if batch_size < 2:
        # batch_size = 1なら計算する目的が消滅する
        is_execute_flag = False
    
    if not is_above_limit:
        # 解像度が低い場合、いくつかの統計値に対する信頼性が著しく低下する可能性がある
        
        if mode == "pool" and pool_num == 1:
            # この条件ならば、平均色をbatch比較する意義があるので、skipしたくない
            pass
        elif mode=="ch_vector" or mode=="pixel" or mode=="ch_sparsity":
            # これらのモードであればピクセル同士の比較であるため、解像度に関係なく使用可能
            pass          
        else:
            is_execute_flag = False
     
    if not is_execute_flag:
        return torch.zeros(1, device=_device, dtype=_dtype)
        
    def extract_features(x, mode):
        # 空間情報の抽出：統計量を測定し、特徴を際立たせる
        
        # (B, C, H, W) -> (B, C, H*W) : チャンネルごとの画素平坦化
        x_flat = x.flatten(2)
        
        features = []

        if mode=="pixel":
            # ToDo: ピクセルそのものを比較するという観点では、高ノイズ領域では。なぜならばbatchごとにノイズが異なるため。
            # ノイズ問題さえなければ原理的には最強なので、解決するまでは残しておきたい。代替機能の導入を検討したいところ
            
            features = [x_flat]
            
            boost   = 1.0
            norm    = area_latents
            
        elif mode=="pool":
            # 各領域のベクトルの平均値を比較する
            # 注：    輝度meanは実質ここで評価されているので、ほかのloss_batchでは比較するべきではない            
            #        7分割のような細かな分割設計にしたい場合は、必ず3分割などを中継して、レイアウトを学ばせること。
            #       3x3だけだと物足りないように感じるかもしれないが、ほとんどの画像で3x3を学べれば、実質細かい部分もわかる
            #       5x5を追加することで、高周波側へ逃がしてしまう力を弱める。
            #       7x7以上はスライス跡が目立つのでやめるべき。縦長or横長では分割困難になる
                        
            pool_x = adaptive_avg_pool2d_for_latents(x.float(), (pool_num, pool_num))
                                 
            features = [
                pool_x.flatten(1), 
            ]
            
            boost   = 0.01 # 基本的にpoolによるbatch比較は、評価する領域が粗いせいか、gradがやたら大きくなりやすく、等倍だと支配的になりすぎる
            norm    = pool_num ** 2
            
        elif mode=="ch_vector":
            # loss_ch_vectorと類似機能
            # poolよりもピクセル単位で細かく比較できるので、キャプション差を捉えやすい（たとえば、人の顔は狭い区間にたくさんのタグがあるため）

            x_vector = get_ch_vector(x_flat)

            features = [x_vector]
            
            boost   = 1.0
            norm    = area_latents           

        elif mode=="ch_sparsity":   

            eps=1e-16
            
            x_l1 = torch.abs(x).sum(dim=1)            
            x_l2 = torch.sqrt(torch.pow(x, 2).sum(dim=1))     

            x_l1 = torch.clamp(x_l1, min=eps)
            x_l2 = torch.clamp(x_l2, min=eps)
            
            x_sparsity = torch.log(x_l1) - torch.log(x_l2)
            
            features = [
                x_sparsity.flatten(1),
            ]
            
            boost   = 1.0
            norm    = area_latents
            
        elif mode=="others": 
            # 基本的には使わない、過去の統計値
            #mean = torch.mean(x, dim=(2, 3))  # 全体的な色味や明るさのトーン → 画像が茶色くなる原因かもしれない、ボツ
            #amax = top_k_mean(largest=True)   # 最も強い光（ハイライト）の勢い → timestep1000付近で4隅欠損するのでボツ
            #amin = top_k_mean(largest=False)  # 最も深い影（ヌケ）の沈み込み → timestep1000付近で4隅欠損するのでボツ
            #std  = torch.std(x, dim=(2, 3))   # 描き込みの密度や質感の激しさ
            features.extend([
                #mean,
                #amax, 
                #amin,
                #std,
            ])
            boost = 1.0
            norm    = area_latents
            
        return torch.cat(features, dim=1), boost, norm        
    
    # snrが同等の全batchペア走査
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            # SNRの差が0.1未満か判定
            snr_diff = torch.abs(_current_snr_weight[i] - _current_snr_weight[j])
            
            if snr_diff < 0.1:
                indices = [i, j]
                #print_storage("keep", f"calc loss_batch_relation\tbatch{i} vs batch{j}")
                
                # 特徴抽出と標準化を一括処理
                feat_pred, boost, norm  = extract_features(noise_pred[indices], mode)
                feat_target, _, norm    = extract_features(target[indices], mode)

                # batch方向のベクトル化
                # 【参考】 以前は、２つのbatchのペアのL1距離で計算していたが、
                # 差が小さいときのアンダーフローや勾配爆発に悩まされたので、差をベクトルとして扱うことにした。ベクトル化するほうがシンプルかつ軽量
                feat_pred      = get_batch_vector(feat_pred)
                feat_target    = get_batch_vector(feat_target)
                                
                # 補正
                # boost : 学習効果を調整する効果
                # norm  : 要素数などによる正規化
                scales = boost / norm
                feat_pred   = feat_pred.mul(scales)
                feat_target = feat_target.mul(scales)

                loss = train_util.conditional_loss(
                    feat_pred.float(),
                    feat_target.float(),
                    reduction="none",
                    loss_type=args.loss_type,
                    huber_c=huber_c
                )
                
                total_loss_relation += loss.sum()

    return total_loss_relation
    
#-----------------------------------------

_LOSS_CONFIG = {
    #名称と、(重み倍率, gamma, deadband, [カテゴリ, 役割])
    # gamma     ：lossを何乗するか。大きいほど学習後期よりも学習序盤に寄与する。
    # deadband  ：この閾値以下のlossをカットして、過適合を防ぐ
    # カテゴリ      :lossの種類のカテゴリを示す。同一カテゴリであることの識別であるため、名前そのものには意味がない
    # 役割        ：同一カテゴリ内の処理を行う際、どれをbaseとするかの判定に使用する
    "base   ":  (1.0, 1.0, 0.0, [None, None]),    # 最も大切なlossではあるが、grad/loss効率が低いので、強調したいところ
    "pool_3x": (1.0, 1.0, 0.0, ["pool", "base"]),
    "pool_5x": (1.0, 1.0, 0.0, ["pool", "sub"]),
    "ch_vector": (1.0, 1.0, 0.01, [None, None]),
    "ch_flow_r2":  (1.0, 1.0, 0.0, [None, None]),
    "sparsity":  (1.0, 1.0, 0.0, [None, None]),
    "pair_128px": (1.0, 1.0, 0.0, ["pair", "base"]),
    "pair_64px": (1.0, 1.0, 0.0, ["pair", "sub"]),
    "pair_32px": (1.0, 1.0, 0.0, ["pair", "sub"]),
    "batch_p_3x": (1.0, 1.0, 0.0, ["batch_pool", "base"]),
    "batch_p_5x": (1.0, 1.0, 0.0, ["batch_pool", "sub"]),
    "batch_px": (1.0, 1.0, 0.0, [None, None]),
    "batch_ch_vec": (1.0, 1.0, 0.0, [None, None]),
    "batch_spars": (1.0, 1.0, 0.0, [None, None]),    
}

_LOSS_NAMES = list(_LOSS_CONFIG.keys())


_loss_EMA_dict = {}
    
def combine_losses_dynamically(
    losses_list: list[torch.Tensor], 
    global_step,
    area_img,
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
    valid_grad_indices = []  # losses_listのインデックスのうち、grads_listに入ったもの
    base_shape_tensor = None # 基準となる形状（通常は最初の有効なロス）を特定するための準備
    scalar_only_sum = torch.tensor(0.0, device=_device) # PCgradで合成できない、たまに発生するスカラー
    
    
    # 各損失の勾配算出とリスト化 ------------------------------------------------------------
    for i, item in enumerate(losses_list):
        if i < len(_LOSS_NAMES) and item is not None:
            
            loss_value_raw, pred = item
            loss_name = _LOSS_NAMES[i]
            
            # 各lossの個別整形 -----------------------------------------------
            
            static_weight, gamma_value, deadband, _ = _LOSS_CONFIG.get(loss_name, (1.0, 1.0, 0.0, [None, None]))
            
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
            loss_instance = torch.pow(loss_instance + 1e-16, all_gamma) # **= all_gammaと同義だが、NaN対策として必須
            loss_instance *= all_weight
                        
            def reduce_micro_loss_on_lowres(loss, area_img, loss_abs_mean_ema):
                """
                ・解像度依存の勾配減衰 
                    1344pxを基準とした不感帯の適用
                    loss通常,reso小  ・・・減衰小,  loss小,reso通常  ・・・減衰,  loss小,reso小   ・・・減衰大
                    効果：低解像度画像に対する過適合を防ぐ。パレート解の高解像度寄りに寄せる。低解像度画像はハズレ値の学習に専念させる。
                ・なぜgradではなくlossに適用したのか。→ 低解像度における高いgradは許容できないノイズである可能性がある。lossという目線でフィルタリングすべき
                ・loss_base :これを超えるかどうかで減衰しやすさが変わる。
                """
                
                ideal_reso = 1024 # 理想的な解像度。この解像度以上であれば減衰ほぼなし                  
                resolution = math.sqrt(area_img) # px単位での解像度
                
                if resolution >= ideal_reso:    # 補正不要なので、そのまま返す
                    return loss
                    
                def _cutoff_weight(loss_rate, threshold, sharpness=10, threshold_offset=0.1):
                    """
                    シグモイド関数を用いた、急峻なゲート処理
                    loss_rate: loss / loss_abs_mean_ema (lossの倍率)
                    threshold: (ideal_reso - resolution) / ideal_reso (解像度に基づく閾値)
                    """ 
                    
                    # torch.sigmoid(x) = 1 / (1 + exp(-x))
                    weight = torch.sigmoid(sharpness * (loss_rate - threshold + threshold_offset))
                    weight = torch.clamp(weight, min=0.05) # アンダーフローによる勾配断絶対策
                    
                    return weight
                   
                # 解像度が低いほど大きくなるペナルティ要素
                threshold = max(1.0 - (resolution / ideal_reso), 0.0)
                
                loss_rate = loss.div(loss_abs_mean_ema + 1e-10)
                weight = _cutoff_weight(loss_rate, threshold)
                
                return loss * weight
            
            # EMA（指数移動平均）の更新

            current_loss_instance_mean = loss_instance.mean().item()
            if current_loss_instance_mean > 0.0: #主にloss計算がスキップされたケースではEMA計算してはいけないためスキップ
                if loss_name not in _loss_EMA_dict:
                    _loss_EMA_dict[loss_name] = current_loss_instance_mean
                else:
                    beta = 0.99
                    _loss_EMA_dict[loss_name] = (beta * _loss_EMA_dict[loss_name]) + ((1.0 - beta) * current_loss_instance_mean)
                
                loss_instance = reduce_micro_loss_on_lowres(loss_instance, area_img, _loss_EMA_dict[loss_name])
            
            if global_step % print_interval_step == 0 or is_debug_mode:
                def loss_bar(loss):
                    max_bar = 10  # 0.05刻みで最大0.5 → 10段階
                    capped_loss = min(loss, 0.5)
                    blocks = int(capped_loss / 0.05)
                    bar = "█" * blocks + " " * (max_bar - blocks)
                    return bar
                bar = loss_bar(current_loss_instance_mean)
                
                if i == 0:
                    print_storage("keep", f"\n loss_name\tgamma\t\tSt_weight\tDy_weight\tloss補正前/補正後")
                print_storage("keep", f" {loss_name} \t{base_gamma}*{gamma_value}\t\t{static_weight:.3f}\t\t{dynamic_weight:.3f}\t\t{current_loss_mean.item():.3f}/{current_loss_instance_mean:.3f}\t|{bar}|")
            
            loss_scalar = loss_instance.mean()
            
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
                
                if global_step % print_interval_step == 1 or is_debug_mode_grad:                    
                    if grad.any():
                        # 0の除く最小絶対値を使用。基本的にgradの絶対最小値は０なので、０除外しないと役立たない
                        abs_min_val = grad[grad != 0].abs().min().item()
                    else:
                        abs_min_val = 0.0
                    
                    if i == 0:
                        print_storage("keep", f"\n grad_abs\tmax\t\tmin\t\tmean")
                    print_storage("keep", f" {loss_name} \t{grad.abs().max().item():.2e}\t{abs_min_val:.2e}\t{grad.abs().mean().item():.2e}")

                # grad clipping (緊急時向け)                
                # grad.clamp_(-1e-5, 1e-5) # SDXL向けの緊急時専用ブレーキ
                                  
                if is_logging_grad:
                    grad_logger.log_gradient({
                        "step": global_step,
                        "name": loss_name.strip(),
                        "max": grad.abs().max().item(),
                        "mean": grad.abs().mean().item()
                    })

            else:
                # グラフがつながっていない場合は、形状を合わせたゼロ勾配を代入
                grad = torch.zeros_like(pred).detach()
                
            grads_list.append(grad)
            valid_grad_indices.append(i)
            
            if base_shape_tensor is None:
                base_shape_tensor = loss_value_raw

    if not grads_list:
        return torch.tensor(0.0, device=losses_list[0].device, requires_grad=True)

    # Multi-task Learning（MTL）実行前のgradの処理 -------------------
        
    # 相互にgradを操作できるように、形状を揃える
    # 二重ループ外で形状情報を整理し、最大要素数でパディングを適用
    max_numel = max(g.numel() for g in grads_list)
    org_grads = []
    original_shapes = []
    for g in grads_list:
        original_shapes.append(g.shape)
        g_f = g.reshape(-1)
        
        if g_f.numel() < max_numel:
            g_f = torch.nn.functional.pad(g_f, (0, max_numel - g_f.numel()))
            
        g_f = torch.nan_to_num(g_f, nan=0.0, posinf=0.0, neginf=0.0) # 計算前に nan/inf を 0 に置換（安全装置） 
        org_grads.append(g_f)

    edited_grads_temp = [g.clone() for g in org_grads]

    def _grad_orthogonalization(indices, org_grads, edited_grads_temp, is_same_category = False):
        """
        gradの直交化を行い、重複成分を除去する
        広義ではMTLの操作であり、狭義では、PCgradの思想を使用して独自アレンジしている。
       
        indices                 :lossのリストインデックス     
        org_grads               :平坦化したgradのオリジナル。直交処理のinput
        edited_grads_temp       :平坦化したgradのclone。直交処理のoutput
        is_same_category   :比較対象が同一カテゴリかどうか。Trueなら同一
        """
        
        # デバッグ専用統計値
        conflict_count = 0
        total_reduction_norm = 0.0
        original_norms_sum = 0.0

        # PCgradの適用  ---------------------------------------------

        # 同一カテゴリ内の処理（Duplicate除去）の場合、役割に基づいてソート
        if is_same_category:
            def get_priority(idx):
                name = _LOSS_NAMES[valid_grad_indices[idx]]
                role = _LOSS_CONFIG[name][3][1]
                return 0 if role == "base" else 1
            
            # indices自体を、baseが先頭に来るよう並び替える
            indices.sort(key=get_priority)
        else:
            random.shuffle(indices) # 評価順をシャッフルして、loss並び順の影響を計算結果に与えにくくする            

        for i in indices:
            gi_flat = edited_grads_temp[i]
            
            for j in indices:
                if i == j: continue
                
                if is_same_category:
                    role_i = _LOSS_CONFIG[_LOSS_NAMES[valid_grad_indices[i]]][3][1]
                    role_j = _LOSS_CONFIG[_LOSS_NAMES[valid_grad_indices[j]]][3][1]
                    # subをbaseで削る組み合わせ以外はすべてスキップ
                    if not (role_i == "sub" and role_j == "base"):
                        continue

                if is_debug_mode_PCgrad:
                    if is_same_category:
                        name_i = _LOSS_NAMES[valid_grad_indices[i]]
                        name_j = _LOSS_NAMES[valid_grad_indices[j]]
                        role_i = _LOSS_CONFIG[name_i][3][1]
                        role_j = _LOSS_CONFIG[name_j][3][1]
                        print_storage("keep", f" [Check] {name_i}({role_i}) vs {name_j}({role_j})")
                
                gj_flat = org_grads[j] # 比較対象は事前平坦化済みテンソル
                
                if gi_flat.dtype != gj_flat.dtype:
                    gj_flat = gj_flat.to(gi_flat.dtype)

                dot_prod = torch.dot(gi_flat, gj_flat)

                if is_debug_mode_PCgrad:
                    before_norm = torch.norm(gi_flat) # 修正前のノルムを記録

                if dot_prod > 0 or is_same_category:
                    # 単純加算による「二重の押し込み」を防ぎ、強い方の要求を優先するモード
                    # 適用条件：同方向（オーバーシュート対策）または、同一カテゴリモード

                    if is_same_category:
                        my_role = _LOSS_CONFIG[_LOSS_NAMES[valid_grad_indices[i]]][3][1]
                        target_role = _LOSS_CONFIG[_LOSS_NAMES[valid_grad_indices[j]]][3][1]
                        if my_role == "base" and target_role == "sub":
                            continue
                    
                    gi_flat = torch.where(
                        torch.abs(gi_flat) > torch.abs(gj_flat),
                        gi_flat,
                        gj_flat
                    )
                    
                elif dot_prod < 0:
                    # 直交成分だけを残し、干渉成分を相殺するモード
                    # 適用条件：異方向（干渉対策）

                    conflict_count += 1
                    
                    # mag_sq（勾配の大きさの二乗）が極端に小さい場合に発生する不定形演算を回避                   
                    mag_sq = torch.dot(gj_flat, gj_flat)
                    if mag_sq > 1e-10:
                        # 補正係数の絶対値を最大1.0に制限し、勾配の爆発を阻止
                        conflict_ratio = torch.clamp(dot_prod / mag_sq, min=-1.0, max=1.0)
                        gi_flat = gi_flat - conflict_ratio * gj_flat

                else:
                    # 直交、またはどちらかの勾配が0の場合は干渉も重複もないためスキップ
                    pass
                    
                if is_debug_mode_PCgrad:                    
                    # 修正後のノルムとの差分から「カットされた量」を蓄積
                    after_norm = torch.norm(gi_flat)
                    total_reduction_norm += abs((before_norm - after_norm).item())
                    original_norms_sum += before_norm.item()

            # 修正済み平坦化テンソルを格納
            edited_grads_temp[i] = gi_flat

        if is_debug_mode_PCgrad:            
            # PCGrad 統計値の計算
            reduction_rate = (total_reduction_norm / (original_norms_sum + 1e-10)) * 100
            print_storage("keep", f" [PCGrad Stats] Conflicts(+-unmatch): {conflict_count} | Grad Cut Rate: {reduction_rate:.2f}%")
        
        return edited_grads_temp

    # grad直交化 ---------------------------------
    
    # 同一カテゴリ同士の直交化（重複除去）
    
    # カテゴリ分類
    cat_group = {}
    for list_idx, loss_idx in enumerate(valid_grad_indices):
        loss_name = _LOSS_NAMES[loss_idx]
        cat_name = _LOSS_CONFIG[loss_name][3][0]
        if cat_name is not None:
            if cat_name not in cat_group:
                cat_group[cat_name] = []
            cat_group[cat_name].append(list_idx)      

    # 各カテゴリ間で直交化
    for cat_name, indices_category in cat_group.items():
        if len(indices_category) > 1:
            edited_grads_temp = _grad_orthogonalization(
                indices_category, 
                org_grads, 
                edited_grads_temp, 
                is_same_category=True
            )
    org_grads = [g.clone() for g in edited_grads_temp] # カテゴリ計算の結果を反映

    # 全gradに対して直交化
    indices_full = list(range(len(grads_list)))
    #indices_full = [k for k in range(len(grads_list)) if "base" not in _LOSS_NAMES[valid_grad_indices[k]]] # 【ボツ案】baseのみ対象外とする場合。base単体の学習効果が100step時点で明らかに失われ、ディティール学習が困難になる

    edited_grads_temp = _grad_orthogonalization(
        indices_full, 
        org_grads,
        edited_grads_temp, 
        is_same_category = False
    )

    # 平坦化したテンソルを元の形状に復元して edited_grads を作成 --------------------------
    edited_grads = []
    for k, efg in enumerate(edited_grads_temp):
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
    accumulated_grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    # 勾配を注入するためのアンカー（通常はnoise_pred）を抽出
    grad_anchor = next(item[1] for item in losses_list if item is not None)

    # 勾配のすり替え
    # total_loss_tensorをdetachして元の勾配を切り離し、PCGrad済みの勾配（grad_anchor経由）を合成
    all_loss = total_loss_tensor.detach()
    all_loss += (accumulated_grad * (grad_anchor - grad_anchor.detach())).sum()
    all_loss += scalar_only_sum
                   
    return all_loss

def get_loss_all(
    loss_base, 
    target, 
    noise_pred, 
    args, 
    huber_c,
):
    
    global _random_seed_1, _dtype, _device
    # 生成用ランダムseed 
    _random_seed_1 = random.randint(0, 2**32 - 1)
    
    # 縮小処理をする場合の下限面積[元画像サイズベースのpx単位]
    area_lower_limit_img        = 512 ** 2 
    area_lower_limit_latents    = area_lower_limit_img  // (8**2)
    
    H, W, area_latents, _ = get_image_hw(target)
    
    is_above_limit = area_latents >= area_lower_limit_latents      

    # 面積スケーリング係数（1024px=128latent を 1.0 とする）
    # 低解像度画像が優位になりすぎてしまうlossに対して影響度を下げる
    reso_scale = min(1.0, math.sqrt((H * W) / (128 * 128)))
    #print(f"reso_scale\t{reso_scale}") for debug
        
    # 各lossの算出=====================================

    _dtype = target.dtype
    _device = target.device

    # バッチ次元がない場合は追加 (C, H, W) -> (1, C, H, W)
    is_batched = (target.dim() == 4)
    target_mod = target if is_batched else target.unsqueeze(0)
    pred_mod = noise_pred if is_batched else noise_pred.unsqueeze(0)
    
    # 過適合の心配があるloss向けに、gaussianフィルタをかける
    # 適用条件： poolを使用しない関数であること（あるならば適用しても意味がない）。
    # ピクセル比較を重要視しているloss(使用したら存在意義を失う)。
    # いくら上記を満たそうとも、ピクセル比較系lossが多すぎると、過適合になる。そのため、一部のピクセル系lossは妥協して適用させるべき
    #target_gaus    = filtering_gaussian(target_mod)
    #pred_gaus      = filtering_gaussian(pred_mod)  
    
    loss_pool_3x, loss_pool_5x = [
        calc_loss_pool(
            target_mod, pred_mod, args, huber_c, is_above_limit=True, pool_num=n
        )
        for n in [3, 5]
    ]
    
    loss_ch_vector = calc_loss_ch_vector(target_mod, pred_mod, args, huber_c)
    
    loss_ch_flow_r2 = calc_loss_ch_flow_2(
            target_mod, pred_mod, args, huber_c, is_above_limit, searching_radius = 2.0
    )
    
    loss_sparsity = calc_loss_sparsity(target_mod, pred_mod, args, huber_c)
    
    loss_pair_corr_128px, loss_pair_corr_64px, loss_pair_corr_32px = [
        calc_loss_pair_correlation(
            target_mod, pred_mod, args, huber_c, is_above_limit, scale_px=s
        )
        for s in [128, 64, 32]
    ]
    
    loss_batch_pool_3x, loss_batch_pool_5x = [
        calc_loss_batch_relation(
            target_mod, pred_mod, args, huber_c, area_latents, is_above_limit,  mode="pool", pool_num=n,
        )
        for n in [3, 5]
    ]

    loss_batch_pixel = calc_loss_batch_relation(
        target_mod, pred_mod, args, huber_c, area_latents, is_above_limit=True, mode="pixel",
    )
    
    loss_batch_ch_vector = calc_loss_batch_relation(
        target_mod, pred_mod, args, huber_c, area_latents, is_above_limit=True, mode="ch_vector",
    )
    
    loss_batch_sparsity = calc_loss_batch_relation(
        target_mod, pred_mod, args, huber_c, area_latents, is_above_limit=True, mode="ch_sparsity",
    )    
   
    # 統合するlossをリスト化する。
    # リストの位置が重要なので、必ず何かを代入すること。統合をスキップしたい場合はNoneを代入する。
    all_computed_losses = [
        loss_base,
        loss_pool_3x,
        loss_pool_5x,
        loss_ch_vector,
        loss_ch_flow_r2,
        loss_sparsity,
        loss_pair_corr_128px,
        loss_pair_corr_64px,
        loss_pair_corr_32px,
        loss_batch_pool_3x,
        loss_batch_pool_5x, 
        loss_batch_pixel,       
        loss_batch_ch_vector,
        loss_batch_sparsity,
    ]
    
    # NaN/Inf補正およびnoise_predとのペアリング
    all_computed_losses = [
        (torch.nan_to_num(l, nan=0.0, posinf=0.0, neginf=0.0), noise_pred) 
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
    snr_weight_view, # 高ノイズ領域のときほど値が大きくなる係数（0～1.0）
    current_mask=None,
):
    global _current_snr_weight, _current_mask
    _current_snr_weight = snr_weight_view
    _current_mask = current_mask
        
    # loss値の各種計算
    all_computed_losses = get_loss_all(loss, target, noise_pred, args, huber_c)        
    
    # ロスの集計と重み付け
    _, _, _, area_img = get_image_hw(target)
    loss = combine_losses_dynamically(all_computed_losses, global_step, area_img)

    if is_print_screen:
        print_storage("print")

    # VRAM解放のため、個々の損失テンソルを削除
    for l in all_computed_losses:
        if l is not None:
            del l
    
    # 不要な変数を削除
    del all_computed_losses
    
    return loss
