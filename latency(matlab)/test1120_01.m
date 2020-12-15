% singal���� event�� ��������
signal_data = ALLEEG(1).data;
origin_event = ALLEEG(1).event;

% ä�� 3���� signal data�� ��������
chan3_latency_list = zeros(3,271);
 for i=1:length(origin_event)
%     fprintf("%d\n",origin_event(i).latency);
%   fprintf("%f\n",signal_data(1:3,origin_event(i).latency));
    chan3_latency_list(:,i) = signal_data(1:3,origin_event(i).latency);
 end
 
 % ���� latency �ε��� �迭 ����
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
% ������ latency �ε��� �迭 ����
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
 % reject latency �ε��� �迭 ����
reject_index = zeros(1,60);%����� ������ 60�����ٴ� �۴�.
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
 
 %left���� reject trial ����
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
 %right���� reject trial ����
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

% ��ü �������� chan3_latency_list�κ��� ���ʿ� �ش��ϴ� �κ�
left_signal = zeros(3,0); %�켱 ���̰� ��� �������� ���� 0
left_signal_cnt=1;
for j=1:length(left_index) %60
     if(left_index(j)==271) %������ trial(271��°)�� ������ ��� ����ó��
        for k=origin_event(left_index(j)).latency:length(signal_data)
            left_signal(:,left_signal_cnt) = signal_data(1:3,k);
            left_signal_cnt = left_signal_cnt+1;
        end
        break;
     end
    if(left_index(j)==0)
        continue;
    end
    for k=origin_event(left_index(j)).latency:origin_event(left_index(j)+1).latency %���� latency�� ��Ÿ�� ������ �ڸ���  
        left_signal(:,left_signal_cnt) = signal_data(1:3,k);
         filename=sprintf('left_trial%d',j);
         save(filename,'left_signal');
         left_signal_cnt = left_signal_cnt+1;
    end
end
% ��ü �������� chan3_latency_list�κ��� �����ʿ� �ش��ϴ� �κ�
right_signal = zeros(3,0); %�켱 ���̰� ��� �������� ���� 0
right_signal_cnt=1;
for j=1:length(right_index) %60
    if(right_index(j)==271) %������ trial(271��°)�� �������� ��� ����ó��
        for k=origin_event(right_index(j)).latency:length(signal_data)
            right_signal(:,right_signal_cnt) = signal_data(1:3,k);
            right_signal_cnt = right_signal_cnt+1;
        end
        break;
    end
    if(right_index(j)==0)
        continue;
    end
    for k=origin_event(right_index(j)).latency:origin_event(right_index(j)+1).latency %���� latency�� ��Ÿ�� ������ �ڸ���
         right_signal(:,right_signal_cnt) = signal_data(1:3,k);
         filename=sprintf('right_trial%d',j);
         save(filename,'right_signal');
         right_signal_cnt = right_signal_cnt+1;
    end
end