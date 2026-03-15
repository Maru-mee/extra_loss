各種学習スクリプトに対して、新しいlossを追加するスクリプトです。<br>
sd-scriptsの非公式MODのような位置づけです。<br>

## 目次
* [役割](#役割)
* [特徴](#特徴)
* [対象](#対象)
* [インストール方法](#インストール方法)
* [使用方法](#使用方法)
* [補足](#補足)
* [ライセンス/License](#ライセンスlicense)
* [Disclaimer / 免責事項](#disclaimer--免責事項)
* [issue, PRについて](#issue-prについて)

## 役割
* 画像生成AIに対する学習効率の向上。
* 従来のloss mse/L1では捉えきれない、画像の構造を詳しくlossへ変換する<br>
* それっぽい似ている絵ではなく、さらに具体的な特徴を検出する<br>
* ベースモデルからの大規模学習における、素早い学習をアシストする<br>

## 特徴
* 計算時間は無視できる(0.1sec/step未満目安)<br>
* VRAM増加は軽微（目安として0.1GBより十分小さい）<br>
* 従来の学習ツールへの拡張のしやすさ<br>
* latentの特性を考慮して最適化したloss群
* PCgrad思想を利用<br>
  * 多数のlossが混在していても、必要なgrad成分だけを抽出し、grad過大によるオーバーシュートを防ぐ<br>
  * 効果の弱いlossをカットオフし、計算速度を向上<br>

## 対象
* SDXLで検証済み。FLUX.1やAnimaといったモデルでも原理的には使用可能と考えています。
* 解像度512px未満の画像ではいくつかのloss計算がskipされます。低解像度のノイズを拾わないようにするため。
 したがって、512px未満の画像は残しても害はありませんが、効果を体感しにくくなります。
* ε-pred, v-predに対応。
* タグの正しさを学ぶ力が強いため、キャプションはそこそこしっかり作成しておいたほうが無難です。
  * 比較的データセットの弱点が健在化しやすくなるためです。
* alpha_maskには非対応。技術的には可能ですが実装が面倒なので省略。

## インストール方法
お手持ちのloss計算直後に、lossをcalc_extra_losses関数で上書きしてください<br>
添付の記入例を参考にしてください<br>

* 学習ツール（sd-scriptsなど）のフォルダ内に、任意のフォルダを作成（フォルダ名の例：custom）<br>
* customフォルダへ、loss_extra_calc.pyを保存<br>
* customフォルダへ__init__.pyという、空のファイルを追加<br>
* 自分が使いたい学習ツール（例：sdxl_train.pyなど）へ、下記を追加<br>


  * 文頭に下記を追加<br>
  ```
  from custom.loss_extra_calc import calc_extra_losses
  　（customは作成したフォルダ名）
  ```
  * loss = train_util.conditional_lossの直後付近へ、<br>
    呼び出し文loss = calc_extra_losses(以下略）の呼び出し文を追加<br>
    * つまり、通常のloss計算直後に挿入すればOKです。
    * 注意点として、必ず勾配断絶しない位置に挿入してください
    * たとえば、loss.meanが実行された後に挿入しても、勾配追跡情報は消失するため十分な効果は得られません。
    * loss補正系（例：min_SNR_gammaやdebiased estimation）よりも前に配置してください
 * もし、ライブラリ不足していた場合は、個別にインストールしてください。sd-scriptsが動作している場合は、おそらくすべてインストール済みと思います。

## 使用方法
通常通り学習を実施します。
* loss_typeはL1から試すことをお勧めします。なぜならば、L2だとハズレ値を優先して学習するリスクがあるためです。
代替案として、smooth_l1及びc<0.2程度でもいいでしょう。
   
## 補足
* もし、効きが強過ぎると感じた場合は、_LOSS_CONFIGの重み倍率を調整してみてください
* このlossは比較的厳密な学習を求めることを目的に検証しています。ベースモデルの情報を維持したい場合には、やや強すぎるかもしれません。

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
* 無名のスクリプトですので、気楽にやりましょう。
