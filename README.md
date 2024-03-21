# 28th_study

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## []()

### [민웅](/민웅.py)

```py

```

### [상미](//상미.py)

```py

```

### [성구](//성구.py)

```py

```

### [영준](//영준.py)

```py

```

<br/>

</details>

<br/><br/>

# 지난주 스터디 문제

<details markdown="1">
<summary>접기/펼치기</summary>

## [행렬 곱셈 순서](https://www.codetree.ai/problems/matrix-multiplication-order/description)

### [민웅](./행렬%20곱셈%20순서/민웅.py)

```py
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
```

### [상미](./행렬%20곱셈%20순서/상미.py)

```py

```

### [성구](./행렬%20곱셈%20순서/성구.py)

```py

```

### [영준](./행렬%20곱셈%20순서/영준.py)

```py
# MCM(Matrix Chain Multipilication 알고리즘)
# 행렬을 곱할 때 MCM으로 결합법칙을 어떻게 적용할 지 찾고, 그 결과대로 곱하는게 평균시간복잡도가 낮다고 함
# 보통은 결합 위치(k)를 저장하지 않고 곱셈의 최소 횟수까지만 확인하는 문제만 나옴.
N = int(input())
A = [0]*(N+1)   # 행열크기
for i in range(N):
    A[i], A[i+1] = map(int, input().split())  # 행렬 크기 저장

D = [[0]*(N+1) for _ in range(N+1)]            # Dij : Ai부터 Aj까지 최소 곱셈횟수

for l in range(1, N):                          # 곱하는 행렬의 개수
    for i in range(1, N-l+1):                  # i 곱하는 맨 앞 행렬
        j = i+l                                # j 맨 마지막 행렬 
        min_v = 1000000000
        for k in range(i, j):                  # (Ai...Ak)(Ak+1...Aj) 결합법칙 적용하는 왼쪽 괄호의 끝 행렬번호
            min_v = min(min_v, D[i][k]+D[k+1][j]+A[i-1]*A[k]*A[j])     # 결합 위치를 바꿨을 때 최소 곱셈 횟수 갱신
        D[i][j] = min_v                        #  Ai...Aj까지 최소 곱셈 횟수

print(D[1][N])
```

## [nxm 표 이동](https://www.codetree.ai/problems/move-n-x-m-table-9/description)

### [민웅](./nxm%20표%20이동%20/민웅.py)

```py
import sys
input = sys.stdin.readline
from collections import deque
dxy = [(0, 1), (0, -1), (1, 0), (-1, 0)]

N, M = map(int, input().split())

field = [list(map(int, input().split())) for _ in range(N)]
visited = [[[0, 0] for _ in range(M)] for _ in range(N)]

q = deque()
# i, j, cnt, 벽통과
if field[0][0] == 1:
    q.append([0, 0, 1, 1])
    visited[0][0][1] = 1
else:
    q.append([0, 0, 1, 0])
    visited[0][0][0] = 0

ans = float('inf')
while q:
    x, y, cnt, one = q.popleft()
    if x == N-1 and y == M-1:
        if cnt < ans:
            ans = cnt
        break
    
    for d in dxy:
        nx = x + d[0]
        ny = y + d[1]

        if 0 <= nx <= N-1 and 0 <= ny <= M-1:
            if field[nx][ny] == 0:
                if not visited[nx][ny][one]:
                    q.append([nx, ny, cnt+1, one])
                    visited[nx][ny][one] = cnt+1
            else:
                if not one and not visited[nx][ny][1]:
                    q.append([nx, ny, cnt+1, 1])
                    visited[nx][ny][1] = cnt+1

if ans == float('inf'):
    print(-1)
else:
    print(ans)
```

### [상미](./nxm%20표%20이동%20/상미.py)

```py

```

### [성구](./nxm%20표%20이동%20/성구.py)

```py

```

### [영준](./nxm%20표%20이동%20/영준.py)

```py

```

## [코드트리 음식점](https://www.codetree.ai/problems/codetree-restaurant/description)

### [민웅](./코드트리%20음식점/민웅.py)

```py
import sys
input = sys.stdin.readline

N, K = map(int, input().split())
menus = {}
ans = -1
for _ in range(N):
    food = int(input())
    if food in menus.keys():
        menus[food] += 1
    else:
        menus[food] = 1
    
    if menus[food] >= K:
        if ans == -1:
            ans = food
        else:
            if food < ans:
                ans = food
    
    print(ans)
```

### [상미](./코드트리%20음식점/상미.py)

```py

```

### [성구](./코드트리%20음식점/성구.py)

```py

```

### [영준](./코드트리%20음식점/영준.py)

```py

```

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

</details>
