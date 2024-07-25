模板

```c++
struct TopK {
    int K;
    // st1 保存前 K 小值 st2 保存其他值
    multiset<LL> st1, st2;
    // sum 维护前 st1 中元素和
    LL sum;
    
    TopK(int K): K(K), sum(0) {}
    
    // 调整 st1 和 st2 的大小，保证调整后 st1 保存前 K 小值
    void adjust()
    {
        while (st1.size() < K && st2.size() > 0)
        {
            LL t = *(st2.begin());
            st1.insert(t);
            sum += t;
            st2.erase(st2.begin());
        }
        while (st1.size() > K)
        {
            LL t = *prev(st1.end());
            st2.insert(t);
            st1.erase(prev(st1.end()));
            sum -= t;
        }
    }
    
    // 插入
    void add(LL x)
    {
        if (st2.size() && x >= *(st2.begin())) st2.insert(x);
        else st1.insert(x), sum += x;
        adjust();
    }
    
    // 删除
    void del(LL x)
    {
        auto it = st1.find(x);
        if (it != st1.end()) st1.erase(it), sum -= x;
        else st2.erase(st2.find(x));
        adjust();
    }
};
```

---

题目参见 [100178. 将数组分成最小总代价的子数组 II](https://leetcode.cn/problems/divide-an-array-into-subarrays-with-minimum-cost-ii/)

