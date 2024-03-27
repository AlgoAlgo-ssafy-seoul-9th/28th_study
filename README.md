# 28th_study

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [멀티탭 스케줄링](https://www.acmicpc.net/problem/1700)

### [민웅](/멀티탭%20스케줄링/민웅.py)

```py
# 1700_멀티탭 스케줄링_multitap scheduling
import sys
input = sys.stdin.readline

N, K = map(int, input().split())
appliances = list(map(int, input().split()))

# 사용중 기기
using_app = set()
cnt = 0
ans = 0

for i in range(K):
    tmp = appliances[i]
    if appliances[i] in using_app:
        continue
    else:
        if cnt != N:
            using_app.add(tmp)
            cnt += 1
        else:
            # 사용중 기기 복사본
            tmp_set = set(k for k in using_app)
            for j in range(i+1, K):
                # 우선순위가 가장 낮은 기기 한개면 stop
                if len(tmp_set) == 1:
                    break
                if appliances[j] in tmp_set:
                    tmp_set.remove(appliances[j])

            change = tmp_set.pop()
            using_app.remove(change)
            using_app.add(tmp)
            ans += 1

print(ans)

# 더 빠른? 코드
"""
# 1700_멀티탭 스케줄링_multitap scheduling
import sys
# import time
input = sys.stdin.readline
# start_time = time.time()

N, K = map(int, input().split())
appliances = list(map(int, input().split()))

using_app = set()
cnt = 0
ans = 0

next_use = {}

for i in range(K):
    if appliances[i] not in next_use.keys():
        next_use[appliances[i]] = []
    else:
        next_use[appliances[i]].append(i)


for i in range(K):
    tmp = appliances[i]
    if appliances[i] in using_app:
        next_use[tmp].pop(0)
        continue
    else:
        if cnt != N:
            using_app.add(tmp)
            cnt += 1
        else:
            change = [-1, -1]
            for s in using_app:
                if not next_use[s]:
                    using_app.remove(s)
                    using_app.add(tmp)
                    break
                else:
                    if next_use[s][0] > change[1]:
                        change = [s, next_use[s][0]]
            else:
                next_use[change[0]].pop(0)
                using_app.remove(change[0])
                using_app.add(tmp)
            ans += 1

print(ans)
# end_time = time.time()
# execution_time = end_time - start_time
# print(execution_time)
"""
```

### [상미](/멀티탭%20스케줄링/상미.py)

```py
import sys
input = sys.stdin.readline

N, K = map(int, input().split())
lst = list(map(int, input().split()))
tmp = []
cnt = 0
for l in range(K):
    if len(tmp) < N and lst[l] not in tmp:    # 멀티탭 빈 곳 있으면
        tmp.append(lst[l])
        continue

    if lst[l] in tmp:        # 이미 해당 코드 꽂혀 있으면
        continue
    else:
        most_far_num = 0
        max_dist = -1
        for plug in tmp:
            rest = lst[l+1:]
            if plug in rest:        # 남은 코드 번호 중 제일 멀리 있는 걸 뻄
                if rest.index(plug) > max_dist:
                    max_dist = rest.index(plug)
                    most_far_num = plug
            else:           # 더 이상 이 코드 안 나오면
                max_dist = 101
                most_far_num = plug
        tmp.remove(most_far_num)
        cnt += 1
        tmp.append(lst[l])
print(cnt)
```

### [성구](/멀티탭%20스케줄링/성구.py)

```py
# 1700 멀티탭 스케줄링
import sys
input = sys.stdin.readline


def main():
    N, K = map(int, input().split())
    plug = [0] * N  # 플러그
    que = tuple(map(int, input().split()))
    cnt = 0  # 뽑기 횟수
    idx = 0  # plug 인덱스
    for i in range(K):
        if que[i] in plug:  # 이미 있는 플러그는 패스
            continue
        if idx < N:     # 빈자리가 있으면 채우기
            plug[idx] = que[i]
            idx += 1
        else:
            # 다음 사용이 가장 늦은 아이 뽑기
            used = [(0,l) for l in range(N)]
            for j in range(N):
                for k in range(i+1, K):
                    if plug[j] == que[k]:
                        used[j] = (k, j)
                        break
            used.sort()
            # 다음 사용이 없으면 그냥 뽑아도 상관없음
            if used[0][0]:
                # 다음 사용이 늦은 아이 교체
                plug[used[-1][1]] = que[i]
            else:
                plug[used[0][1]] = que[i]
            cnt += 1
    print(cnt)

if __name__ == "__main__":
    main()

```

### [영준](/멀티탭%20스케줄링/영준.py)

```py
N, K = map(int, input().split())
elec = list(map(int, input().split()))
multi = [0]*(K+1)           # 실제 크기가 아니라 카운트배열로 선언
elec_cnt = 0                # 현재 연결된 기기 수
ans = 0                     # 기기를 뽑은 횟수
arr = [[0]*(K+1) for _ in range(K+1)]   # 얼마나 뒤에 사용되는지 표시
for j in range(K-1, 0, -1):
    for i in range(1, K+1):
        if i == elec[j]:
            arr[i][j-1] = 1
        elif arr[i][j]:
            arr[i][j-1] = arr[i][j] + 1

for j in range(K):
    if elec_cnt<N and multi[elec[j]]==0:    # 남은 자리가 있고, 처음 사용되는 기기면
        elec_cnt += 1
        multi[elec[j]] = 1
    elif elec_cnt==N and multi[elec[j]]==0: # 자리가 없고, 새로운 기기면, 가장 나중에 사용될 기기를 제거
        max_idx = 0
        for i in range(1, K+1):
            if multi[i] and arr[i][j]==0:   # 다시 사용하지 않는 기기면 제거
                max_idx = i
                break
            if multi[i] and arr[max_idx][j] < arr[i][j]:
                max_idx = i
        multi[max_idx] = 0
        multi[elec[j]] = 1
        ans += 1
print(ans)
```

<br/>

## [책 나눠주기](https://www.acmicpc.net/problem/9576)

### [민웅](/책%20나눠주기/민웅.py)

```py
# 9576_책나눠주기_Handind out books
import sys
import heapq
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    hq = []
    for _ in range(M):
        s, g = map(int, input().split())
        heapq.heappush(hq, [g, s])
    books = [0]*(N+1)
    ans = 0
    while hq:
        a, b = heapq.heappop(hq)
        for i in range(b, a+1):
            if not books[i]:
                ans += 1
                books[i] = 1
                break
    print(ans)
```

### [상미](/책%20나눠주기/상미.py)

```py
import sys
input = sys.stdin.readline

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    taken = [0] * (N+1)
    stu = []
    for _ in range(M):
        a, b = map(int, input().split())
        stu.append((a, b))
    stu.sort(key = lambda x:(x[1], x[0]))  # b 기준 정렬

    for (a, b) in stu:
        for i in range(a, b+1):
            if not taken[i]:
                taken[i] = 1
                break

    print(taken.count(1))

```

### [성구](/책%20나눠주기/성구.py)

```py
# 9576 책 나눠주기
import sys
input = sys.stdin.readline


def main():
    _, M = map(int, input().split())
    orders = sorted(list(tuple(map(int, input().split())) for _ in range(M)), key=lambda x:(x[1], x[0]))
    visited = set()
    cnt = 0
    for i in range(M):
        for j in range(orders[i][0]-1, orders[i][1]):
            if j in visited:
                continue
            visited.add(j)
            cnt += 1
            break
    print(cnt)

    return


if __name__ == "__main__":
    for _ in range(int(input())):
        main()
```

### [영준](/책%20나눠주기/영준.py)

```py

```

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
# A1x...AN까지 행렬을 곱할 때 최소 곱셈 횟수를 구하는 알고리즘
# 행렬을 곱할 때 MCM으로 결합법칙을 어떻게 적용할 지 찾고, 그 결과대로 곱하는게 평균시간복잡도가 낮다고 함
# 보통은 결합 위치(k)를 저장하지 않고 곱셈의 최소 횟수를 찾는 문제만 나옴.
N = int(input())
A = [0]*(N+1)   # 행열크기
for i in range(N):
    A[i], A[i+1] = map(int, input().split())  # 행렬 크기 저장

D = [[0]*(N+1) for _ in range(N+1)]            # Dij : Ai부터 Aj까지 최소 곱셈횟수

for l in range(1, N):                          # 곱셈 횟수
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
import sys
from collections import deque
input = sys.stdin.readline

N, M = map(int, input().split())

field = tuple(tuple(map(int, input().split())) for _ in range(N))

def bfs():
    que = deque([(0,0,field[0][0])])
    visited = [[[0,0] for _ in range(M)] for _ in range(N)]
    visited[0][0][field[0][0]] = 1
    while que:
        i, j, is_visit = que.popleft()
        if i == N-1 and j == M-1:
            return visited[i][j][is_visit]

        for di, dj in [(1,0), (0,1), (-1,0), (0,-1)]:
            ni,nj = i+di, j+dj
            if 0 <= ni < N and 0 <= nj < M:
                if not visited[ni][nj][is_visit]:
                    if is_visit:
                        if not field[ni][nj]:
                            visited[ni][nj][1] = visited[i][j][1] + 1
                            que.append((ni,nj,1))
                    else:
                        visited[ni][nj][field[ni][nj]] = visited[i][j][0] + 1
                        que.append((ni,nj,field[ni][nj]))
    return -1

print(bfs())
```

### [영준](./nxm%20표%20이동%20/영준.py)

```py
# 난 왜 이렇게 복잡하게...
from collections import deque

def bfs(i, j, visited):
    q = deque()
    q.append((i,j))
    #visited= [[0]*m for _ in range(n)]
    visited[i][j] = 1
    while q:
        i, j = q.popleft()

        for di, dj in [[0,1],[1,0],[0,-1],[-1,0]]:
            ni, nj = i+di, j+dj
            if 0<=ni<n and 0<=nj<m and arr[ni][nj]==0 and visited[ni][nj]==0:
                q.append((ni, nj))
                visited[ni][nj] = visited[i][j] + 1

di = [0,1,0,-1]
dj = [1,0,-1,0]


n, m = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(n)]
visited1 = [[0]*m for _ in range(n)]
visited2 = [[0]*m for _ in range(n)]

flag = 2
if arr[0][0]+arr[n-1][m-1] == 1: # 출발 또는 도착이 벽이면
    arr[0][0] = arr[n-1][m-1] = 0
    flag = 1
elif arr[0][0]+arr[n-1][m-1] == 2: # 모두 1이면 이동 불가
    flag = 0
min_v = 1000000
if flag:
    bfs(0, 0, visited1)         # 좌상단 시작
    if visited1[n-1][m-1] != 0:
        min_v = visited1[n-1][m-1]
    if flag==2:
        bfs(0, 0, visited1)  # 좌상단 시작
        bfs(n-1, m-1, visited2)     # 우하단 시작


        if visited1[n-1][m-1] != 0:     # 우하단 도착 가능한 경우
            miv_v = visited1[n-1][m-1]      # 기둥을 그대로 둔 최소길이
        for i in range(n):
            for j in range(m):
                if arr[i][j]==1:    # 벽을 사이에 두고 좌상단, 우하단 시작이 만나고 (같은자리 x) 벽이 없으면 최소가 되는 경우를 찾기
                    for k in range(4):
                        for l in range(4):
                            if k != l:
                                ki, kj = i+di[k], j+dj[k]
                                li, lj = i+di[l], j+dj[l]
                                if 0<=ki<n and 0<=kj<m and 0<=li<n and 0<=lj<m and arr[ki][kj]+arr[li][lj]==0:
                                    if visited1[ki][kj]*visited2[li][lj] != 0: # 탐색가능한 칸에 한해
                                        if min_v > visited1[ki][kj] + visited2[li][lj] + 1:
                                            min_v = visited1[ki][kj] + visited2[li][lj] + 1

if min_v==1000000:
    min_v = -1
print(min_v)
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
```

### [영준](./코드트리%20음식점/영준.py)

```py

```

</details>

<br/><br/>

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

## Belady's Min Algorithm

### [페이지 교체 알고리즘](https://steady-coding.tistory.com/526)

</details>
