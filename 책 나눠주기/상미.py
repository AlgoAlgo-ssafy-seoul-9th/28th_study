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
