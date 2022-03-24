# HW4: Wine dataset classification by kohonen network
1. Importing library \
![](https://i.imgur.com/5g8FSfa.png) \
以上為引用的library，csv用來讀csv檔，numpy用來計算矩陣，sklearn用來分train和testing資料，tensorflow為model的整體架構。
2. Reading & splitting dataset \
![](https://i.imgur.com/7EqOmUM.png) \
這個部分是先將”wine.csv”讀入reader這個x這個list，然後再把list中所有最後的一項(結果，ex: 1 or 2 or 3)傳給y。 
3. kohonen network \
在kohonen network的架構中，共有四個函式。
  * __init__ \
  ![](https://i.imgur.com/fvGMje6.png) \
  在類別初始化的時候，定義n*m個神經元，alpha為learning rate，dim為資料的維度，weight_vec為代表神經元的向量(隨機0~1的數值)，最後weight_label為training完後weight所代表的label。
  * training_operator \
    ![](https://i.imgur.com/3ycNCL7.png) \
    training_operator為在training時要更新weight的function，在這裡我們採用 @tf.function來加速整個流程。我們將每個input跟weight取最相似值，再更新其weight值。首先我們取的Best Matching Unit (bmu)，可以知道當次輸入的weight哪個與input最相似，再利用取的的bmu乘上alpha，即可知道本次要更新的是哪個weight以及其值。而下面兩個for迴圈是為了要normalize我們的weight值。這裡我們先以column normalize再以row做normalize，對column normalize的目的是為了不讓feature中特別大的值稀釋要其他feature的值。
  * train \
    ![](https://i.imgur.com/NGeJvJD.png) \
  首先，前兩個for迴圈先normalize input值。再來，我們讓每個iteration中的每個input vector帶入training_operator，以此答要更新weight的目的。最後，再將weight找到與他最相似的input vector的label為其weight_label。
  * test \
    ![](https://i.imgur.com/UXxGy1T.png) \
	最後，我們將testing資料代入並取test vector與weight vector最相似者，若是testing label與 weight_label一樣則為正確，這樣我們就可以得到正確率。
  
# Conclusion
在這裡我們利用tensorflow的架構訓練kohonen network，並使用了tensorflow中的graph mode來加速計算，而kohonen架構原理為更新winner神經元來使神經元逐漸clustering。若我們想要實現classification的話，只要將資料與神經元比較，最相似者則與其同類別。

