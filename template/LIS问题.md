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

