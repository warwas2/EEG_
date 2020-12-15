% singal값과 event값 가져오기
signal_data = ALLEEG(1).data;
origin_event = ALLEEG(1).event;

% 채널 3개의 signal data만 가져오기
chan3_latency_list = zeros(3,271);
 for i=1:length(origin_event)
%     fprintf("%d\n",origin_event(i).latency);
%   fprintf("%f\n",signal_data(1:3,origin_event(i).latency));
    chan3_latency_list(:,i) = signal_data(1:3,origin_event(i).latency);
 end
 
 % 왼쪽 latency 인덱스 배열 저장
 left_index = zeros(1,60);
 before = 1;
 left_index(1)=left_args2(1);
 for i=2:length(left_args2)
     if isnan(left_args2(i))
         continue;
     end
     if left_index(before)~=left_args2(i)
        left_index(1,before+1) = left_args2(i);
        before = before + 1;
     end
 end
% 오른쪽 latency 인덱스 배열 저장
 right_index = zeros(1,60);
before=1;
right_index(1)=right_args2(1);
 for i=2:length(right_args2)
     if isnan(right_args2(i))
         continue;
     end
     if right_index(before)~=right_args2(i)
        right_index(1,before+1) = right_args2(i);
        before = before + 1;
     end
 end
 % reject latency 인덱스 배열 저장
reject_index = zeros(1,60);%몇개인지 모르지만 60개보다는 작다.
before=1;
reject_index(1)=reject_args2(1);
 for i=2:length(reject_args2)
     if isnan(reject_args2(i))
         continue;
     end
     if reject_index(before)~=reject_args2(i)
        reject_index(1,before+1) = reject_args2(i);
        before = before + 1;
     end
 end
 
 %left에서 reject trial 제거
for j=1:length(reject_index)
    if(reject_index(j)==0)
        break;
    end
    for i=1:length(left_index)
        if(reject_index(j)+1==left_index(i))
            left_index(i)=0;
        end
    end
end
 %right에서 reject trial 제거
for j=1:length(reject_index)
    if(reject_index(j)==0)
        break;
    end
    for i=1:length(right_index)
        if(reject_index(j)+1==right_index(i))
            right_index(i)=0;
        end
    end
end

% 전체 데이터인 chan3_latency_list로부터 왼쪽에 해당하는 부분
left_signal = zeros(3,0); %우선 길이가 어느 정도인지 몰라서 0
left_signal_cnt=1;
for j=1:length(left_index) %60
     if(left_index(j)==271) %마지막 trial(271번째)이 왼쪽인 경우 예외처리
        for k=origin_event(left_index(j)).latency:length(signal_data)
            left_signal(:,left_signal_cnt) = signal_data(1:3,k);
            left_signal_cnt = left_signal_cnt+1;
        end
        break;
     end
    if(left_index(j)==0)
        continue;
    end
    for k=origin_event(left_index(j)).latency:origin_event(left_index(j)+1).latency %같은 latency가 나타날 때까지 자르기  
        left_signal(:,left_signal_cnt) = signal_data(1:3,k);
         filename=sprintf('left_trial%d',j);
         save(filename,'left_signal');
         left_signal_cnt = left_signal_cnt+1;
    end
end
% 전체 데이터인 chan3_latency_list로부터 오른쪽에 해당하는 부분
right_signal = zeros(3,0); %우선 길이가 어느 정도인지 몰라서 0
right_signal_cnt=1;
for j=1:length(right_index) %60
    if(right_index(j)==271) %마지막 trial(271번째)이 오른쪽인 경우 예외처리
        for k=origin_event(right_index(j)).latency:length(signal_data)
            right_signal(:,right_signal_cnt) = signal_data(1:3,k);
            right_signal_cnt = right_signal_cnt+1;
        end
        break;
    end
    if(right_index(j)==0)
        continue;
    end
    for k=origin_event(right_index(j)).latency:origin_event(right_index(j)+1).latency %같은 latency가 나타날 때까지 자르기
         right_signal(:,right_signal_cnt) = signal_data(1:3,k);
         filename=sprintf('right_trial%d',j);
         save(filename,'right_signal');
         right_signal_cnt = right_signal_cnt+1;
    end
end