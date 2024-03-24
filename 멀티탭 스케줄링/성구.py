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
    