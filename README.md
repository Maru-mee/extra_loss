各種学習スクリプトに対して、新しいlossを追加するスクリプトです。<br>
sd-scriptsのオプション機能です。<br>

役割
* 画像生成AIに対する学習効率の向上。
* 従来のloss mse/L1では捉えきれない、画像の構造を詳しくlossへ変換する<br>
* それっぽい似ている絵ではなく、さらに具体的な特徴を検出する<br>
* ベースモデルからの大規模学習における、素早い学習をアシストする<br>

特徴<br>
* 計算速度が無視できるほど小さい<br>
* VRAM増加は軽微<br>
* 従来の学習ツールへの拡張のしやすさ<br>
* latentの特性を考慮して最適化したloss群
* PCgrad思想を利用<br>
  * 多数のlossが混在していても、必要なgrad成分だけを抽出し、grad過大によるオーバーシュートを防ぐ<br>
  * 効果の弱いlossをカットオフし、計算速度を向上<br>
* ε-pred, v-predに対応<br>

・使い方<br>
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
    通常のloss計算直後に挿入する、という意味です。<br>
 * もし、ライブラリ不足していた場合は、個別にインストールしてください。sd-scriptsが動作している場合は、おそらくすべてインストール済みと思います。
   
補足
* もし、効きが強過ぎると感じた場合は、_LOSS_CONFIGの重み倍率を調整してみてください
* このlossは比較的厳密な学習を求めることを目的に検証しています。ベースモデルの情報を維持したい場合には、やや強すぎるかもしれません。
