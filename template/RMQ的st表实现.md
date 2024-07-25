# RMQ的st表实现

原理见 https://www.luogu.com.cn/problem/P3865

模板很简单，就是倍增思想

```c++
int rmq[N][21], lg[N], n, m;

int query(int l, int r) {
    int p = lg[r - l + 1];
    return max(rmq[l][p], rmq[r - (1 << p) + 1][p]);
}

	cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> rmq[i][0];
        if (i == 1) {
            lg[i] = 0;
        } else {
            lg[i] = lg[i >> 1] + 1;
        }
    }
    for (int p = 1, len = 2; p <= 20; p++, len <<= 1) {
        for (int i = 1; i + len - 1 <= n; i++) {
            rmq[i][p] = max(rmq[i][p - 1], rmq[i + len / 2][p - 1]);
        }
    }
    while (m -- ) {
        int x, y;
        cin >> x >> y;
        cout << query(x, y) << endl;
    } 
```

应用见：[3117. 划分数组得到最小的值之和](https://leetcode.cn/problems/minimum-sum-of-values-by-dividing-array/)

