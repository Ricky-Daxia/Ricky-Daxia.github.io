看一道典题，如何转化为 LIS

>   给两个 1-n 的排列，问这两个排列的 LCS

![LIS](https://cdn.jsdelivr.net/gh/Ricky-Daxia/Hei_Xiu/202407252306136.png)

LIS 模板

```c++
class Solution {
public:
    int lengthOfLIS(vector<int> &nums) {
        vector<int> g;
        for (int x : nums) {
            auto it = ranges::lower_bound(g, x);
            if (it == g.end()) {
                g.push_back(x); // >=x 的 g[j] 不存在
            } else {
                *it = x;
            }
        }
        return g.size();
    }
};
```

其实是贪心+二分的思想，用 `f[i]` 表示长为 `i` 的 LIS 结尾元素的最大值

1.  `f[i]` 随着 `i` 单调增，这一点可以由反证法证明
2.  据此可以在 `f` 中二分查找 `a[i]` 的插入点

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) { // 最长严格递增子序列
        int n = nums.size();
        vector<int> q(n + 10, 0);

        int len = 0;
        q[0] = -2e9;
        for (int i = 0; i < n; i++)
        {
            int l = 0, r = len;
            while (l < r)
            {
                int mid = l + r + 1 >> 1;
                if (q[mid] < nums[i]) l = mid; // 如果是非递减 就写成 <=
                else r = mid - 1;
            }
            len = max(len, l + 1);
            q[l + 1] = nums[i]; 
        }
        return len;
    }
};
```

还需要掌握**最长下降子序列、最长非增子序列、最长非降子序列**，在于灵活运用 `f[0]` 和 `lower_bound` 和 `upper_bound`

```cpp
int LIS(vector<int> &a) { // 另一种写法
    int n = a.size();
    int f[n + 10];
    int len = 0;
    f[0] = -2e9;
    for (int i = 0; i < n; i++) {
        if (f[len] < a[i]) f[++ len] = a[i]; // 插入尾部的条件
        else *lower_bound(f + 1, f + len + 1, a[i]) = a[i];
    }
    return len;
}

int LNAS(vector<int> &a) { // 最长不上升子序列
    int n = a.size();
    int f[n + 10];
    int len = 0;
    f[0] = 2e9;
    for (int i = 0; i < n; i++) {
        if (f[len] >= a[i]) f[++ len] = a[i];
        else *upper_bound(f + 1, f + len + 1, a[i], greater<int>()) = a[i];
    }
    return len;
}
```

树上 LIS 问题怎么做？求出从根节点到每个节点的 LIS 长度，直接在 DFS 过程中用二分求 LIS 做法做即可，注意要**恢复现场**。题目见 [ABC165F](https://atcoder.jp/contests/abc165/tasks/abc165_f)

~~LIS 的两种二分写法~~

```c++
// version 1
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) 
    {
        scanf("%d", &a[i]);
        p[a[i]] = i;
    }
    for (int i = 1; i <= n; i++) 
    {
        scanf("%d", &b[i]);
        b[i] = p[b[i]];
    }
    int len = 0;
    memset(f, 0x3f, sizeof f);
    f[0] = 0;
    for (int i = 1; i <= n; i++)
    {
        if (b[i] > f[len]) f[++ len] = b[i];
        else
        {
            int l = 0, r = len;
            while (l < r)
            {
                int mid = l + r >> 1;
                if (f[mid] > b[i]) r = mid;
                else l = mid + 1;
            }
            f[l] = min(b[i], f[l]);
        }
    }
    printf("%d\n", len);
}
// version 2
int main()
{ 
    scanf("%d", &n);
    for (int i = 0; i < n; i++) 
    {
        scanf("%d", &a[i]);
        p[a[i]] = i;
    }
    for (int i = 0; i < n; i++) 
    {
        scanf("%d", &b[i]);
        b[i] = p[b[i]];
    }
    int len = 0;
    f[0] = -2e9;
    for (int i = 0; i < n; i++)
    {
        int l = 0, r = len;
        while (l < r)
        {
            int mid = l + r + 1 >> 1;
            if (f[mid] < b[i]) l = mid;
            else r = mid - 1;
        }
        len = max(len, l + 1);
        f[l + 1] = b[i];
    }
    printf("%d\n", len);
}
```

