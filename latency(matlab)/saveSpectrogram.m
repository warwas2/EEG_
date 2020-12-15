% 1. 1���� gdf���Ͽ��� trial�� ����, ������ ������ ��������
% left_trial = all_left_data{1,2};
% right_trial = all_right_data{1,2};

% 2. ��� trial�� ���� ó��
for tindex=1:60
    current_left = all_left_data{tindex,2};
    current_right = all_right_data{tindex,2};
    
    % rejected trial�� �ƴ� ��쿡�� spectrogram ����
    if ~isempty(current_left)
        left_trial = current_left;
        
        % �� ä�� ���� 1���� �̹��� ����
        % �� trial1���� 3���� �̹��� ����
        % trial 1������ ä�� 1���� �ش��ϴ� signal
        
        for cindex=1:3
           
            % ����
            channel_byTrial = left_trial(:,cindex);
            channel_byTrial = cell2mat(channel_byTrial);%�Ϲ� �迭

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
            
            % ���� ����
            filename=sprintf('B0201T_left%d_chan0%d.png',tindex,cindex);
            saveas(ax, filename);
        end
    end
    
    if ~isempty(current_right)
        right_trial = current_right;
        
        for cindex=1:3
        % ������
            Rchannel_byTrial = right_trial(:,cindex);
            Rchannel_byTrial = cell2mat(Rchannel_byTrial);%�Ϲ� �迭

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
            
            filename=sprintf('B0201T_right%d_chan0%d.png',tindex,cindex);
            saveas(ax, filename);
        end
    end
end
