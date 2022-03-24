# HW1: MAP classifier for the UCI-WINE dataset
1. Importing library \
![](https://i.imgur.com/ligq1jD.png)\
以上為引用的library，csv用來讀csv檔，numpy用來計算矩陣，random用來產生亂數，sklearn用來分train和testing資料。
2. Reading & splitting dataset\
![](https://i.imgur.com/Tkkxw0N.png)\
這個部分是先將”wine.csv”讀入reader這個x這個list，然後再把list中所有最後的一項(結果，ex: 1 or 2 or 3)傳給y，並且從x中刪除y之資訊，這樣就分好x與y的資料。再來我們利用sklearn的train_test_split這個函式以train與test各一半來分成x的train，x的test，y的train，y的test這四個list，接下來我們將其轉換成numpy.array的格式以便後續的矩陣運算。
3. Calculating each component\
![](https://i.imgur.com/Jvy35bc.png)\
首先，我們把training的資料集分成三種，「1」、「2」和「3」，作法是讀取y_train[i]以得知其相對的x_train[i]是屬於1或2還是3。做完之後我們再算出各自的先驗機率，即是這三個物品再這個資料集的機率。我們先將其從list轉換numpy.array，再利用mean和np.cov來計算各自的平均和共變異數矩陣。
4. Calculating each likelihood and its classifier function\
![](https://i.imgur.com/SIjPlqZ.png)\
最後，我們將X_test[i]資料帶入高斯分佈的maximum likelihood公式中，要特別注意的是我們要在comp1、comp2和comp3中找出最小的數值，且也吻合y_test[i]，則我們將correct加一，最後可以得出我們這個Map classifier 的準確率。
![](https://i.imgur.com/ctQGaRH.png)\
## Conclusion
我們在知道資料分佈為高斯分佈時，可以利用貝式分類器中的Map classifier來當分類器。我們先求其每個類別的先驗機率，再算出他的likelihood，結合在一起後找到輸出最小的類別若和該筆資料的類別相符，則分類器預測成功，否則失敗。
