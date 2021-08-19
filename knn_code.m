clc
close all
clear all
load labl.mat
sub_no=100;
save_feat=[];
save_cls=[];
hs=[];
subj=[];
count=0;

%% pcg signal 
   
for D=1:sub_no
     subject_seq = D;
   [ x,fs ,str] = readdataa( D );
    disp('Data reading complete')
    N = length(x); % number of samples
    T = 1/fs; % period
   
   %% downsample
    y=downsample(x,2);
    N1=length(y);
    disp('Downsampling...')
        
    %% 25-400Hz 4th order Butterworth filtering 
          [b,a] = butter(4,[.025 .4],'bandpass');
            z = filter(b,a,y);
          disp('Filtering...')
        
    %% denoising
    level=5;
    wname='sym6';
    zd=wden(z,'rigrsure','s','sln',level,wname);
    disp('Denoising...')
    snr= -20*log10(norm(abs(zd-z))/norm(z));
    
    %% Remove mean
    zd=zd-mean(zd); %remove dc offset
    
   %% Time-Frequency Analysis
   [Xamp,Samp]= Time_freq(zd,fs);
   disp('Time Frequency Analysis');
   
  %% discrete wavelet transform
   N=5;
   [cd,ca] = getdwt(zd); %cd=detail coeff. ca=approximate coeff.
       
   %% feature extraction
        
   [pxx1,f]= pwelch(cd,4,3,500,2000);
   [pxx2,f]= pwelch(ca,4,3,500,2000);   
   %figure(1);  plot(10*log10(pxx1));
   %figure(2);plot(10*log10(pxx2));
   
   f1=wentropy(cd,'shannon');
   f2=wentropy(cd,'log energy');
   f3=wentropy(cd,'threshold',0.2 );
   f4=wentropy(cd,'sure',3 );
   f5=wentropy(cd,'norm',1.1 );
   f6=wentropy(ca,'shannon');
   f7=wentropy(ca,'log energy');
   f8=wentropy(ca,'threshold',0.2 );
   f9=wentropy(ca,'sure',3 );
   f10=wentropy(ca,'norm',1.1 );
    
%% Feature saving

disp('Saving features..')
 feature(D,:) = [ mean(ca) max(ca) min(ca) std(ca) sum(ca.^2) var(ca) skewness(ca) kurtosis(ca)...
      mean(cd) max(cd) min(cd) std(cd) sum(cd.^2) var(cd) skewness(cd) kurtosis(cd)...
       f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 ];   
     [rf cf]=size(feature);
    
end
  % figure; % plot(z); title(str);   
  % figure; subplot(1,2,1); plot(ca(4,:)); title('Approximation coeffs'); 
  % subplot(1,2,2); plot(cd(1,:)); title('Detailed coeffs');
  
    %% 
    HS_feature = feature;
    %group=labl;
    Dtrain=[];
    Dtest=[];
    hs=labl;  
 %% k-fold crossvalidation
    k_fold=20;
    ntree=60;
    mtry=100;%number of predictors sampled for spliting at each node
    d=[];
    indices = crossvalind('Kfold',hs,k_fold);
    cp = classperf(hs)
%tic
pred_all_T = zeros(1,length(hs));
for ii = 1:k_fold 
    disp('trainning...')
    test = (indices == ii); 
    train = ~test; % 1=test set remaining = trainset   
    Dtrain = HS_feature(train,:);
     Dtest = HS_feature(test,:);
       Y = hs(train);
      Y_test = hs(test);
       knnstruct=fitcsvm(Dtrain,Y);     
%          svmstruct=fitcsvm(Dtrain,Y,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
%           'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%           'expected-improvement-plus','ShowPlots',false));
%   CVSVMModel = crossval(svmstruct);
%   classLoss = kfoldLoss(CVSVMModel);
 disp('testing...') 
predd = predict(knnstruct,Dtest);
% L = loss(CMdl,XTest,YTest)
       classperf(cp,predd,test);
       d(ii)=cp.CorrectRate     
        pred_all_T(1,test)=str2double(predd);
      
%       BB = TreeBagger(ntree,Dtrain,Y,'oobpred','on');    
%     figure
%     plot(oobError(BB))
%     xlabel('number of grown trees')
%     ylabel('out-of-bag classification error')
     
%       pred_labl=predict(BB,Dtest);
%     [pred_labl, votes, prediction_per_tree] = class_predict(Dtest,BB);%for C compiler
%        pred_all_T(1,test)=str2double(pred_labl);
%        classperf(cp,pred_labl,test);
%        d(ii)=cp.CorrectRate
%      toc
end

%% Accuracy
k = find(~isnan(d));
d = d(k)
acuracy=mean(d)
ACCURACY = mean(acuracy)
% accu = sum(predd==Y_test)/length(Y_test)*100

%% scatterplot
% 
% figure;
% hgscatter = gscatter(Dtrain(:,15),Dtrain(:,25),Y);
% hold on;
% h_sv=plot(svmstruct.SupportVectors(:,15),svmstruct.SupportVectors(:,25),'ko','markersize',8);

%% Making Confusion Matrix
% 
C_M=zeros(2,2);
 
for ii=1:length(pred_all_T)
          if pred_all_T(ii)==0
              pred_all_T(ii)=1;
          end
     C_M(pred_all_T(ii),hs(ii))=C_M(pred_all_T(ii),hs(ii))+1;
end 
 C_M
 C_M2=[C_M;sum(C_M)]
 C_M3=[C_M2  sum(C_M2')']
 for jj=1:2
     Sensitivity(jj)=C_M3(jj,jj)*100/C_M3(7,jj);
 end
 

%% Making Decisions
% 
% for i=1:2
%     
%     TP(i)=C_M3(i,i);
%     FN(i)=C_M3(7,i)-C_M3(i,i);
%     FP(i)=C_M3(i,7)-C_M3(i,i);
%     TN(i)=C_M3(7,7)-C_M3(i,7)-C_M3(7,i)+C_M3(i,i);
%     
%     ACCURACY(i)=(TP(i)+TN(i))*100/(TP(i)+FN(i)+FP(i)+TN(i));
%     SENSITIVITY(i)=TP(i)*100/(TP(i)+FN(i));
%     PRECISION(i)=TP(i)*100/(FP(i)+TP(i));
%     SPECIFICITY(i)=TN(i)*100/(TN(i)+FP(i));
% end
% 
% over_all_accuracy=trace(C_M)*100/C_M3(7,7)
% toc
