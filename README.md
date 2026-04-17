

各種学習スクリプトに対して、新しいlossを追加するスクリプトです。<br>
sd-scriptsの非公式MOD(非公認MOD)のような位置づけです。<br>

## 目次
<details>
<summary>Click to expand</summary>
 
* [役割](#役割)
* [特徴](#特徴)
* [適用範囲](#適用範囲)
* [インストール方法](#インストール方法)
* [使用方法](#使用方法)
* [補足](#補足)
* [ライセンス/License](#ライセンスlicense)
* [Disclaimer / 免責事項](#disclaimer--免責事項)
* [issue, PRについて](#issue-prについて)

</details>


## 役割
* 画像生成AIに対する学習効率の向上。
* 従来のピクセル単位での比較によるlossでは捉えきれない、画像の構造を詳しくlossへ変換する
  * 現状、L1/MSE lossによるピクセル評価に依存していた場合に効果を体感できます。
  * 従来のピクセル評価における、構造を学ぶ過程で、背景やその他構造を壊してしまうのを防ぐ
* それっぽい似ている絵ではなく、さらに具体的な特徴を検出する
  * 特にタグに対する追従性の向上。
  * UNetによる「よく見かけるテンプレ画像を生成しておけばいいや」という怠惰さをlossとして検出できるようになります。
  * 様々な要素（画風など）が教師画像であるときに、メジャーなデータセットの特徴に偏るのを解消します。
  * マイナータグがどの画像要素なのかを、把握するきっかけを与える。
* ベースモデルからの大規模学習における、素早い学習をアシストする

* 使用例：<br>
<img width="6825" height="2580" alt="ON_OFF比較" src="https://github.com/user-attachments/assets/c1147a33-e044-4a08-88dd-1441b033d125" /><img alt="zko (7)_Target" src="https://github.com/user-attachments/assets/18da6f3e-c8b8-4f0b-ba45-9ce687023631" width="125">

  ポイント：
  * full_body, ポーズ, 画風が、早期に再現及び維持できています。
  * 構造と無関係なノイズ感が低下します。
  * 正則化画像なしにも関わらず、初期の構造破損は最小限に留まり、一貫性のある変化をしています
  * 発色がより鮮やかになります。
  * 上記サンプル画像における学習改善の影響が、データセット全体に対する認識力改善に寄与します。
  * モデル：sd_xl_base_1.0 (Stability_AI)
  * データセット：<https://zunko.jp/con_illust.html>よりお借りしました。
  * 【参考】 図はloss_extra_calc_v0.13を使用
<details><summary>キャプション</summary>
zunko, 1girl, solo, japanese clothes, muneate, tabi, hairband, kimono, smile, open mouth, very long hair, weapon, polearm, short kimono, full body, white_background, standing on one leg, dark green hair, looking at viewer, standing, sandals, simple background, sash, tasuki, :d, obi, naginata, geta --d 341 --s 30 --w 1024 --h 1024 --l 4.0
</details>

## 特徴
* latentsの特性を考慮して最適化したloss群
* latentsを評価するため、軽量
  * 計算時間は無視できる(0.1sec/step未満)
  * VRAM増加は軽微（目安として0.1GBより十分小さい）
* 従来の学習ツールへの拡張のしやすさ
* PCgrad思想を利用
  * 多数のlossが混在していても、必要なgrad成分だけを抽出し、grad過大によるオーバーシュートを防ぐ
  * 効果の弱いlossをカットオフし、計算速度を向上

## 適用範囲
* SDXLで検証済み。FLUX.1やAnimaといったモデルでも原理的には使用可能と考えています。
* ε-pred, v-predに対応。
  * それ以外については、latentsに特化している都合により効果が得にくく、また、snr_weightによるノイズ状況による補正が働かないため、十分な効果は得にくくなります。
* batch_size=2以上を推奨。
  * 1でも機能しますが、batchを活用したlossを活用できなくなります。
* alpha_maskのデータセットには非対応。技術的には可能ですが実装が面倒なので省略。 
* 画像解像度512以上に合わせてチューニング済
  * 解像度512px未満の画像ではいくつかのloss計算がskipされます。低解像度のノイズを拾わないようにするため。
 したがって、512px未満の画像は残しても害はありませんが、効果を体感しにくくなります。
  * そのチューニングがあるために、SD1では効果を得る機会は減ります（一般的に、SD1は700pxを超過する画像を使うと破綻するため）。
  * どうしても512px未満を主体とするデータセット使用したい場合は、手動でコードを書き換えてください。
    area_lower_limit_imgの値を引き下げ、かつ、細かい分割を行ういくつかのlossの機能をOFFにするなど。
* タグの正しさを学ぶ力が強いため、キャプションはそこそこしっかり作成しておいたほうが無難です。
  * 比較的データセットの弱点が健在化しやすくなるためです。

## インストール方法
お手持ちのloss計算直後に、lossをcalc_extra_losses関数で上書きしてください<br>
添付の記入例を参考にしてください<br>

* 学習ツール（sd-scripts）のフォルダ内に、任意のフォルダを作成（フォルダ名の例：custom）
* customフォルダへ、loss_extra_calc.pyを保存
* customフォルダへ__init__.pyという、空のファイルを追加
* 自分が使いたい学習ツール（例：sdxl_train.pyなど）へ、下記を追加


  * 文頭に下記を追加<br>
  ```
  from custom.loss_extra_calc import calc_extra_losses
  　（customは作成したフォルダ名）
  ```
  * loss = train_util.conditional_lossの直後付近へ、<br>
    呼び出し文loss = calc_extra_losses(以下略）の呼び出し文を追加
    * つまり、通常のloss計算直後に挿入すればOKです。
    * 注意点として、必ず勾配断絶しない位置に挿入してください
    * たとえば、loss.meanが実行された後に挿入しても、勾配追跡情報は消失するため十分な効果は得られません。
    * loss補正系（例：min_SNR_gammaやdebiased estimation）よりも前に配置してください
 * もし、ライブラリ不足していた場合は、個別にインストールしてください。sd-scriptsが動作している場合は、おそらくすべてインストール済みと思います。

## 使用方法
通常通り学習を実施します。
* loss_typeはL1から試すことをお勧めします。
  <details><summary>理由</summary>
   
  * なぜならば、L2だとハズレ値を優先して学習しがちで、特定のlossばかりを使用するリスクがあるためです。
  * また、ただでさえ多数のloss(grad)登場により、loss-weight曲面が複雑になっているのに対して、さらにL2にしてしまうとlossの山谷がより急峻になり、局所解からの脱出が極めて困難になります。
  * そうならないように山谷が急にならないよう配慮した設計はしているものの、限度があります。
  * 別の目線では、L2にしなくても、多数のlossによってウェイトをガッチリホールドできているとも言えます。
  * 代替案として、smooth_l1及びc<0.2程度でもいいでしょう。
    
  </details>
* learning rateは既存設定を流用可能です。
  * これまで認知できなかったgradが発生するため、もしかすると、少し下げる必要があるかもしれません。
* optimizerのbeta値は既存設定を流用可能です。
* 下記はおそらく共存可能ですが、本機能を活かす上でなんらかのブレーキが作用する可能性があります
  * debiased_estimation
  * min_snr_gamma
  * multires_noise_iteration
   
## 補足
* もし、効きが強過ぎると感じた場合は、_LOSS_CONFIGの重み倍率を調整してみてください
  * 学習進捗に合わせて機能OFFすることもご検討ください
* このloss群は比較的厳密な学習を求めることを目的に検証しています。ベースモデルの情報を維持したい場合には、やや強すぎるかもしれません。
* このloss群の弱点は、それらの統計的lossへ逃げてしまい、L1/MSEの学習がやや遅れることです。
  * とはいえ、よほど特殊なデータセットでない限り、常時機能ONでの運用を想定しています。

## ライセンス/License
* License: MIT License
* Ownership: This is an original implementation. All code was written by the author. (本コードは作者による自作のオリジナル実装です)
* Technical Reference: The PCGrad algorithm is based on public research. This specific implementation is original. (PCGradアルゴリズムは公知の研究に基づいた実装であり、コード自体は独自に作成されたものです)

## Disclaimer / 免責事項
* Non-Warranty: This software is provided "as is", without warranty of any kind. (本ソフトウェアは現状のまま提供され、いかなる保証もありません)
* Limitation of Liability: In no event shall the author be liable for any claim, damages or other liability arising from the use of this software. (本ソフトウェアの使用により生じたトラブルや損害について、作者は一切の責任を負いかねます)

## issue, PRについて
趣味の範囲、知の共有を目的とした範囲の対応になります。
* issueついては、ある程度、作者の想定使用範囲に合致している場合のみ対応します。
とはいえ、さほど複雑なスクリプトでは有りませんので、
多くの場合、Geminiなどへの質問で解決するはずです。
* pull_requestについても同様です。
なお、新しいlossの追加提案は第三者様の権利侵害が発生しない範囲で、ご自由にどうぞ。
