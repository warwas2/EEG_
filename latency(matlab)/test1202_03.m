% all_left_data�� ù��° trial�� 1�� ä���� signal�� spectrogram �׽�Ʈ

first_left_trial = all_left_data{1,2};
% fprintf(class(first_left_trial));
channel01_byTrial1 = first_left_trial(:,1);%cell
channel01_byTrial1 = cell2mat(channel01_byTrial1);%�Ϲ� �迭
% 
% for i=1:length(channel01_byTrial1)
%     fprintf("%f",channel01_byTrial1(i));%double array
% end

s = spectrogram(channel01_byTrial1);
surf(20*log10(abs(s)));