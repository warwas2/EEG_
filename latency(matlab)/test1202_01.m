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
% �����ֿ� �޸� cell ���� ���

all_left_data = cell(60,2);%�� 60���� trial cell�� ��� ��� �ִ� cell
%all_left_data{60,2}=[];%���� ����

left_signal_cnt=1;
for j=1:length(left_index) %60
% for j=1:1 %1
    all_left_data{j,1}=sprintf('left_trial%d',j);% trial ��ȣ(�̸�) 60�� ����
    
    %�� trial�� signal����
    all_left_data{j,2}=signal_per_trial;
    signal_per_trial=cell(3,0);%trial �� singal data����(����)�� �ٸ��� ������ 0���� ����
    
    %����ó��
    if(left_index(j)==271) %������ trial(271��°)�� ������ ��� ����ó��
        for k=origin_event(left_index(j)).latency:length(signal_data)
%             data{2}(:,left_signal_cnt) = signal_data(1:3,k);
            signal_per_trial{left_signal_cnt,1} = signal_data(1,k);%ù��° ä��
            signal_per_trial{left_signal_cnt,2} = signal_data(2,k);%�ι�° ä��
            signal_per_trial{left_signal_cnt,3} = signal_data(3,k);%����° ä��
            left_signal_cnt = left_signal_cnt+1;
        end
        break;
     end
    if(left_index(j)==0)
        continue;
    end
    
    for k=origin_event(left_index(j)).latency:origin_event(left_index(j)+1).latency %���� latency�� ��Ÿ�� ������ �ڸ���  
%         signal_per_trial{left_signal_cnt} = signal_data(1:3,k);
        signal_per_trial{left_signal_cnt,1} = signal_data(1,k);%ù��° ä��
        signal_per_trial{left_signal_cnt,2} = signal_data(2,k);%�ι�° ä��
        signal_per_trial{left_signal_cnt,3} = signal_data(3,k);%����° ä��
        left_signal_cnt = left_signal_cnt+1;
    end
    
%      if(left_index(j)==271) %������ trial(271��°)�� ������ ��� ����ó��
%         for k=origin_event(left_index(j)).latency:length(signal_data)
%             data{2}(:,left_signal_cnt) = signal_data(1:3,k);
%             left_signal_cnt = left_signal_cnt+1;
%         end
%         break;
%      end
%     if(left_index(j)==0)
%         continue;
%     end
%     for k=origin_event(left_index(j)).latency:origin_event(left_index(j)+1).latency %���� latency�� ��Ÿ�� ������ �ڸ���  
%         data{1} = left_signal_cnt;%trial ��ȣ ����
%         data{2} = signal_data(1:3,k);%signal ����
% 
%         left_signal_cnt = left_signal_cnt+1;
%     end
end