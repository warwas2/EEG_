[start_args1, start_args2, start_args3, start_args4] = eeg_context(ALLEEG(1),{768},{768},1);
[left_args1, left_args2, left_args3, left_args4] = eeg_context(ALLEEG(1),{768},{769},1);
[right_args1, right_args2, right_args3, right_args4] = eeg_context(ALLEEG(1),{768},{770},1);

%1023 ÀÚ¸£±â
[reject_args1, reject_args2, reject_args3, reject_args4] = eeg_context(ALLEEG(1),{768},{1023},1);
