for fileindex=5:16
    [start_args1, start_args2, start_args3, start_args4] = eeg_context(ALLEEG(fileindex),{768},{768},1);
    [left_args1, left_args2, left_args3, left_args4] = eeg_context(ALLEEG(fileindex),{768},{769},1);
    [right_args1, right_args2, right_args3, right_args4] = eeg_context(ALLEEG(fileindex),{768},{770},1);

    %1023 자르기
    [reject_args1, reject_args2, reject_args3, reject_args4] = eeg_context(ALLEEG(fileindex),{768},{1023},1);
    
    pre_fname = ALLEEG(fileindex).setname;
%     fprintf('%s\n',pre_fname);

    % singal값과 event값 가져오기
    signal_data = ALLEEG(fileindex).data;
    origin_event = ALLEEG(fileindex).event;

    % 채널 3개의 signal data만 가져오기
    chan3_length=length(origin_event);
    chan3_latency_list = zeros(3,chan3_length);

     for i=1:length(origin_event)
    %     f printf("%d\n",origin_event(i).latency);
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
    % 저번주와 달리 cell 형태 사용

    all_left_data = cell(60,2);%각 60개의 trial cell을 모두 담고 있는 cell
    %all_left_data{60,2}=[];%위와 동일
    sLength_per_trial=1;%각 trial별 signal 양(길이)

    for j=1:length(left_index) %60
    % for j=1:1 %1로 테스트
        sLength_per_trial=1;

        all_left_data{j,1}=sprintf('left_trial%d',j);% cell에 trial 번호(이름) 1~60 저장

        %각 trial별 signal저장
        signal_per_trial=cell(3,0);%trial 별 singal data개수(길이)가 다르기 때문에 0으로 지정

        %예외처리
        if(left_index(j)==chan3_length) %마지막이 왼쪽인 경우 예외처리
            for k=origin_event(left_index(j)).latency:length(signal_data)
                signal_per_trial{sLength_per_trial,1} = signal_data(1,k);%첫번째 채널
                signal_per_trial{sLength_per_trial,2} = signal_data(2,k);%두번째 채널
                signal_per_trial{sLength_per_trial,3} = signal_data(3,k);%세번째 채널
                sLength_per_trial = sLength_per_trial+1;
            end
            break;
         end
        if(left_index(j)==0)%rejected_trial 건너뛰기
            continue;
        end

        %같은 latency가 나타날 때까지 자르기  
        for k=origin_event(left_index(j)).latency:origin_event(left_index(j)+1).latency 
            signal_per_trial{sLength_per_trial,1} = signal_data(1,k);%첫번째 채널
            signal_per_trial{sLength_per_trial,2} = signal_data(2,k);%두번째 채널
            signal_per_trial{sLength_per_trial,3} = signal_data(3,k);%세번째 채널
            sLength_per_trial = sLength_per_trial+1;
        end

        all_left_data{j,2}=signal_per_trial;%cell에 자른 tiral별 signal 데이터 저장
    end

    all_right_data = cell(60,2);%각 60개의 trial cell을 모두 담고 있는 cell
    %all_left_data{60,2}=[];%위와 동일
    rsLength_per_trial=1;
    for j=1:length(right_index) %60
        rsLength_per_trial=1;
    % for j=1:1 %1
        all_right_data{j,1}=sprintf('right_trial%d',j);% trial 번호(이름) 60개 저장

        %각 trial별 signal저장
        signal_per_trial=cell(3,0);%trial 별 singal data개수(길이)가 다르기 때문에 0으로 지정

        %예외처리
        if(right_index(j)==chan3_length) %마지막이 왼쪽인 경우 예외처리
            for k=origin_event(right_index(j)).latency:length(signal_data)
                signal_per_trial{rsLength_per_trial,1} = signal_data(1,k);%첫번째 채널
                signal_per_trial{rsLength_per_trial,2} = signal_data(2,k);%두번째 채널
                signal_per_trial{rsLength_per_trial,3} = signal_data(3,k);%세번째 채널
                rsLength_per_trial = rsLength_per_trial+1;
            end
            break;
         end
        if(right_index(j)==0)
            continue;
        end

        for k=origin_event(right_index(j)).latency:origin_event(right_index(j)+1).latency %같은 latency가 나타날 때까지 자르기  
            signal_per_trial{rsLength_per_trial,1} = signal_data(1,k);%첫번째 채널
            signal_per_trial{rsLength_per_trial,2} = signal_data(2,k);%두번째 채널
            signal_per_trial{rsLength_per_trial,3} = signal_data(3,k);%세번째 채널
            rsLength_per_trial = rsLength_per_trial+1;
        end

        all_right_data{j,2}=signal_per_trial;
    end
    
    % 2. 모든 trial에 대해 처리
    for tindex=1:60
        current_left = all_left_data{tindex,2};
        current_right = all_right_data{tindex,2};

        % rejected trial이 아닌 경우에만 spectrogram 생성
        if ~isempty(current_left)
            left_trial = current_left;

            % 각 채널 별로 1개씩 이미지 저장
            % 즉 trial1개당 3개씩 이미지 저장
            % trial 1개에서 채널 1개에 해당하는 signal

            for cindex=1:3

                % 왼쪽
                channel_byTrial = left_trial(:,cindex);
                channel_byTrial = cell2mat(channel_byTrial);%일반 배열

                s = spectrogram(channel_byTrial);
                surf(20*log10(abs(s)));
                image(20*log10(abs(s)),'CDataMapping','scaled');
                axis('off');
                ax = gca;
                set(ax,'YDir','normal');
                set(ax,'xtick',[]);
                set(ax,'ytick',[]);
                set(ax, 'units', 'normalized'); %Just making sure it's normalized
                Tight = get(ax, 'TightInset');  %Gives you the bording spacing between plot box and any axis labels
                                                 %[Left Bottom Right Top] spacing
                NewPos = [Tight(1) Tight(2) 1-Tight(1)-Tight(3) 1-Tight(2)-Tight(4)]; %New plot position [X Y W H]
                set(ax, 'Position', NewPos);

                % 파일 저장
                filename=sprintf('%s_left%d_chan0%d.png',pre_fname,tindex,cindex);
                saveas(ax, filename);
            end
        end

        if ~isempty(current_right)
            right_trial = current_right;

            for cindex=1:3
            % 오른쪽
                Rchannel_byTrial = right_trial(:,cindex);
                Rchannel_byTrial = cell2mat(Rchannel_byTrial);%일반 배열

                s = spectrogram(Rchannel_byTrial);
                surf(20*log10(abs(s)));
                image(20*log10(abs(s)),'CDataMapping','scaled');
                axis('off');
                ax = gca;
                set(ax,'YDir','normal');
                set(ax,'xtick',[]);
                set(ax,'ytick',[]);
                set(ax, 'units', 'normalized');
                Tight = get(ax, 'TightInset'); 
                NewPos = [Tight(1) Tight(2) 1-Tight(1)-Tight(3) 1-Tight(2)-Tight(4)];
                set(ax, 'Position', NewPos);

                filename=sprintf('%s_right%d_chan0%d.png',pre_fname,tindex,cindex);
                saveas(ax, filename);
            end
        end
    end
    % 2. 모든 trial에 대해 처리(끝)
end