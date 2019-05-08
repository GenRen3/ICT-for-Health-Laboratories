clear all
close all
clc
%%
rng('default');
tstart=tic;
Kquant=11;% number of quantization levels
Nstates=11;% number of states in the HMM
tolerance=1e-3;
max_iter=200;

ktrain=[1,2,3,4,5,6,7];% indexes of patients for training
ktest=[8,9,10];% indexes of patients for testing
[hq,pq]=pre_process_data(Nstates,Kquant,ktrain);% generate the quantized signals
telapsed = toc(tstart);
disp(['first part, elapsed time ',num2str(telapsed),' s'])
%% HMM training phase....
p=0.9;q=(1-p)/(Nstates-1);
A=[q,p,q*ones(1,Nstates-2)];

for i=1:1:Nstates-1
   A(i+1,:)=circshift(A(i,:),1); %it does the circular matrix
end

A_rand=rand(Nstates,Nstates); %case in which the matrix is random

B=rand(Nstates,Kquant); %emission matrix


for i=1:1:Nstates
    sum_row=sum(B(i,:),2);
    sum_row2=sum(A_rand(i,:),2);
    B(i,:)=B(i,:)./sum_row;
    A_rand(i,:)=A_rand(i,:)./sum_row2;
end

%creation of the Markov Chain
[trans_train_h,emiss_train_h]=hmmtrain(hq(ktrain),A,B,'TOLERANCE',tolerance,'MAXITERATION',max_iter); %healthy
[trans_train_pd,emiss_train_pd]=hmmtrain(pq(ktrain),A,B,'TOLERANCE',tolerance,'MAXITERATION',max_iter); %illed
sensitivity_train=0;
sensitivity_test=0;
specificity_train=0;
specificity_test=0;

for i=1:1:7
    [~,p_false_pos_train(i)]=hmmdecode(hq{i},trans_train_pd,emiss_train_pd);
    [~,p_true_neg_train(i)]=hmmdecode(hq{i},trans_train_h,emiss_train_h);
    
    [~,p_true_pos_train(i)]=hmmdecode(pq{i},trans_train_pd,emiss_train_pd);
    [~,p_false_neg_train(i)]=hmmdecode(pq{i},trans_train_h,emiss_train_h);
    
    if p_true_neg_train(i)>p_false_pos_train(i)
       specificity_train=specificity_train+1;
    end
    
    if p_true_pos_train(i)>p_false_neg_train(i)
       sensitivity_train=sensitivity_train+1;
    end
    
end

specificity_train_perc=(specificity_train/7)*100;

sensitivity_train_perc=(sensitivity_train/7)*100;
fprintf('Specificity training data = %d\n',specificity_train_perc)
fprintf('Patients on training data really healthy = %d/%d\n\n',specificity_train,length(ktrain))
fprintf('Sensitivity training data = %d\n',sensitivity_train_perc)
fprintf('Patients on training data really ill = %d/%d\n',sensitivity_train,length(ktrain))
fprintf('\n')

%% HMM testing phase....
for i=8:10
    [~,p_false_pos_test(i-7)]=hmmdecode(hq{i},trans_train_pd,emiss_train_pd);
    [~,p_true_neg_test(i-7)]=hmmdecode(hq{i},trans_train_h,emiss_train_h);
    
    [~,p_true_pos_test(i-7)]=hmmdecode(pq{i},trans_train_pd,emiss_train_pd);
    [~,p_false_neg_test(i-7)]=hmmdecode(pq{i},trans_train_h,emiss_train_h);
    
    if p_true_neg_test(i-7)>p_false_pos_test(i-7)
       specificity_test=specificity_test+1;
    end
    
    if p_true_pos_test(i-7)>p_false_neg_test(i-7)
       sensitivity_test=sensitivity_test+1;
    end
    
end
specificity_test_perc=(specificity_test/3)*100;
sensitivity_test_perc=(sensitivity_test/3)*100;
fprintf('Specificity test data = %d\n',specificity_test_perc)
fprintf('Patients on test data really healthy = %d/%d\n\n',specificity_test,length(ktest))
fprintf('Sensitivity test data = %d\n',sensitivity_test_perc)
fprintf('Patients on test data really ill = %d/%d\n',sensitivity_test,length(ktest))