import sys
input = sys.stdin.readline

N = int(input())

n_lst = list(map(int, input().split()))

for _ in range(1, N):
    a, b = map(int, input().split())
    n_lst.append(b)

dp = [[0 for _ in range(N)] for _ in range(N)]
ans = 0
if N == 1:
    ans = n_lst[0] * n_lst[1]
else:
    for i in range(2, N + 1):
        for a in range(N - i + 1):
            b = a + i - 1
            dp[a][b] = float('inf')
            for j in range(a, b):
                dp[a][b] = min(dp[a][b], dp[a][j] + dp[j+1][b] + n_lst[a]*n_lst[j+1]*n_lst[b+1])

    ans = dp[0][N-1]
print(ans)