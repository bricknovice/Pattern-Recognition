# HW2: Bhattacharyya boundary for the UCI-WINE dataset
1. Importing library \
![](https://i.imgur.com/xjMasBq.png) \
以上為引用的library，csv用來讀csv檔，numpy用來計算矩陣，random用來產生亂數，sklearn用來分train和testing資料。
2. Reading & splitting dataset \
![](https://i.imgur.com/Ymq1jxp.png) \
這個部分是先將”wine.csv”讀入reader這個x這個list，然後再把list中所有最後的一項(結果，ex: 1 or 2 or 3)傳給y。 
3. Calculating each component \
![](https://i.imgur.com/KdWakg7.png) \
首先，我們把training的資料集分成三種，「1」、「2」和「3」，作法是讀取y[i]以得知其相對的x[i]是屬於1或2還是3。做完之後我們再算出各自的先驗機率，即是這三個物品再這個資料集的機率，並算出各自的mean 和 covariance matrix。
4. Calculating each error probability \
![](https://i.imgur.com/I0WodQv.png) \
我們再將計算Bhattacharyya boundary 的公式套進去，即可得到error probability。\
![](https://i.imgur.com/tO95xg6.png) \
Class1&class2
Class2&class3
Class1&class3
的錯誤機率
# Conclusion 
我們在知道資料分佈為高斯分佈時，可以利用貝式分類器中的Map classifier來當分類器。但我們可以先用Bhattacharyya boundary評估貝式分類器的效果。只要有mean 和covariance matrix即可得到我們要的結果
