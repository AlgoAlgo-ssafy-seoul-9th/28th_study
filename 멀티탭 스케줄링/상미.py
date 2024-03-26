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