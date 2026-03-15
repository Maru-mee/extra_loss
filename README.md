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
* 従来の学習ツールに、１行呼び出し行を追加するだけという容易さ<br>
* PCgrad思想を利用<br>
  * 多数のlossが混在していても、必要なgrad成分だけを抽出し、grad過大によるオーバーシュートを防ぐ<br>
  * 効果の弱いlossをカットオフし、計算速度を向上<br>
* ε-pred, v-predに対応<br>

・使い方<br>
ライブラリとして呼び出せばOKです。<br>
下記はやり方を知らない人向けの説明です<br>

* 学習ツール（sd-scriptsなど）のフォルダ内に、任意のフォルダを作成（フォルダ名の例：custom）<br>
* customフォルダへ、loss_extra_calc.pyを保存<br>
* customフォルダへ__init__.pyという、空のファイルを追加<br>
* 自分が使いたい学習ツール（例：sdxl_train.pyなど）へ、下記を追加<br>
  添付の記入例を参考にしてください<br>

  * 文頭に下記を追加<br>
  ```
  from custom.loss_extra_calc import calc_extra_losses
  　（customは作成したフォルダ名）
  ```
  * loss = train_util.conditional_lossの直後付近へ、<br>
    呼び出し文loss = calc_extra_losses(以下略）の呼び出し文を追加<br>
    通常のloss計算直後に挿入する、という意味です。<br>
