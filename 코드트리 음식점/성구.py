import sys
from collections import defaultdict
input = sys.stdin.readline


N, K = map(int, input().split())
orders = defaultdict(int)
minv = 10**8 +1
orders[minv] = N+1
for _ in range(N):
    menu = int(input())
    orders[menu] += 1
    if orders[menu] < K:
        if orders[minv] == N+1:
            print(-1)
        else:
            print(minv)
    else:
        minv = min(minv, menu)
        print(minv)