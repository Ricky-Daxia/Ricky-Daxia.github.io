# go 代码

变量声明：`var even, odd int`

定义 map：`vis := map[int]bool{}`

定义模数：`const MOD int = 1e9 + 7`

定义无穷：`const inf = math.MaxInt32 - 1`

最大最小值：`math.MinInt` 或者 `math.MaxInt`

判断字符是否为大小写：`unicode.IsUpper(c)` 或者 `unicode.IsLower(c)`，取某个字符用 `rune(s[i])`

abs 函数：`func abs(x int) int { if x < 0 { return -x }; return x }`

字符串不可变，要转成 `s := []byte(S)` 才可以修改，最后用 `string(s)` 变回字符串

想实现 `'a' + k`，写成 `limit := 'a' + byte(k)`

比较序关系，用 `cmp.Compare(a[i-1], a[i]) == cmp.Compare(b[i-1], b[i])`

排序

```go
sort.Slice(idx, func(i, j int) bool {
     return arr[i] < arr[j]
})

slices.SortFunc(items, func(a, b []int) int { return b[0] - a[0] })

sort.Ints(people)
```

字符串转数字、分割、拼接

```go
func discountPrices(sentence string, discount int) string {
    d := 1 - float64(discount) / 100
    a := strings.Split(sentence, " ")
    for i, w := range a {
        if len(w) > 1 && w[0] == '$' {
            price, err := strconv.Atoi(w[1:])
            if err == nil {
                a[i] = fmt.Sprintf("$%.2f", float64(price) * d)
            }
        }
    }
    return strings.Join(a, " ")
}
```

多维数组定义

```go
f := make([][]int, n)
for i := range f {
    f[i] = make([]int, K + 1)
    f[i][0] = 1
}
```

删除 **第一个数字字符** 以及它左边 **最近** 的 **非数字** 字符

```go
func clearDigits(s string) string {
    st := []rune{}
    for _, c := range s {
        if unicode.IsDigit(c) {
            st = st[:len(st) - 1]
        } else {
            st = append(st, c)
        }
    }
    return string(st)
}
```

定义函数

```go
var dfs func(int, int) int
dfs = func(i, j int) (res int) {

}
```

假设序列要拼接一个序列：`append(a, b...)`

查找元素在有序数组的位置：`sort.SearchInts(sorted, x)`

自定义类型

```go
type edge struct{ to, wt int }
g := make([][]edge, n)
for _, e := range edges {
    x, y, wt := e[0], e[1], e[2]
    g[x] = append(g[x], edge{y, wt})
    g[y] = append(g[y], edge{x, wt})
}
```

二分查找：`p := sort.Search(i, func(j int) bool { return events[j][1] >= e[0] })`

pop_count：`i := bits.OnesCount(uint(s))`

返回空数组：`return nil`

字符偏移，先算 ascii 值再转成 byte

```go
		newChar := 'a' + ((int(char) - 'a' + k) % 26)
		bytes[i] = byte(newChar)
```

