% all_left_data의 첫번째 trial의 1번 채널의 signal만 spectrogram 테스트

first_left_trial = all_left_data{1,2};
% fprintf(class(first_left_trial));
channel01_byTrial1 = first_left_trial(:,1);%cell
channel01_byTrial1 = cell2mat(channel01_byTrial1);%일반 배열
% 
% for i=1:length(channel01_byTrial1)
%     fprintf("%f",channel01_byTrial1(i));%double array
% end

s = spectrogram(channel01_byTrial1);
surf(20*log10(abs(s)));