# 9576 책 나눠주기
import sys
input = sys.stdin.readline


def main():
    # N, M = map(int, input().split())
    # books = [[0 for _ in range(N)] for _ in range(M)]
    # orders = sorted(list(tuple(map(int, input().split())) for _ in range(M)), key=lambda x:(x[1], x[0]))
    # 2차원 배열, 사람별 범위 1체크
    # 1~3 : 1 1 1 0
    # 2~4 : 0 1 1 1
    # for i in range(M):
    #     start, end = orders[i]
    #     for j in range(start-1, end):
    #         books[i][j] = 1
    # visited = set()
    # cnt = 0
    # for i in range(N):
    #     for j in range(M):
    #         if j in visited:    # 이미 책을 받은 사람은 패스
    #             continue
        
    #         if books[j][i]:     # 희망 범위에 있으면 책 주고 다음 책 탐색
    #             visited.add(j)
    #             cnt += 1
    #             break
    
    # 간단한 방법
    N, M = map(int, input().split())
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