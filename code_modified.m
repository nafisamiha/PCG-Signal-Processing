clc
close all
clf
tic

save_fea=[];
save_cls=[];
hs1=[];hs2=[];
%% pcg signal 
   
for D=1:7
     subject_no = D;
    [x,fs,str] = readdataa( D );
    %disp('Data reading complete')
    N = length(x); % number of samples
    T = 1/fs; % period
   
   % percent_train=70;

   % [trainData,testData,trainLabel,testLabel] = helperRandomSplit(percent_train,x);
   
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
    z=z-mean(z); %remove dc offset
        
   %  [despiked_signal] = schmidt_spike_removal(original_signal, fs)
   
   %% discrete wavelet transform
   N=5;
   [cd,ca] = getdwt(z);
   %cd=detail coeff. ca=approximate coeff.
   
   
   %% feature extraction
   
    
    [pxx1,f]= pwelch(cd,4,3,500,2000);
    [pxx2,f]= pwelch(ca,4,3,500,2000);
   
   
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
   
   
   
   m=3;
   cls=2; % c: number of classes 
   tau=1;
   scale=4;
       
     feature=[ mean(ca) max(ca) min(ca) std(ca) var(ca) skewness(ca) kurtosis(ca)...
       mean(cd) max(cd) min(cd) std(cd) var(cd) skewness(cd) kurtosis(cd)...
       f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 ];
  
    save_fea=[save_fea;feature];
    save_cls=[save_cls;point];
    
     %[FEATURE_validation] = [feature];
    cnt=0;
     cnt=cnt+1;
     
     if point == 1
         hs1=[hs1;cnt];
     elseif point == -1
         hs2=[hs2;cnt];
     end
     
     end
  % figure;
   % plot(z);title(str);
    
  % figure; subplot(1,2,1), plot(ca(4,:)); title('Approximation coeffs'); 
 %  subplot(1,2,2), plot(cd(1,:)); title('Detailed coeffs');



 %% Random Forest Classification
 
 size(save_fea);
 size(save_cls);
 
 HS_feature=save_fea;
    HS=save_cls;
 
 
 %% k-fold crossvalidation
    k_fold=5;
    ntree=50;
    mtry=100;%number of predictors sampled for spliting at each node
    d=[];
    indices1= crossvalind('Kfold',hs1,k_fold);
    indices2= crossvalind('Kfold',hs2,k_fold);

tic
pred_all_T = zeros(1,length(HS));
cp = classperf(HS);


for ii = 1:k_fold 
    disp('trainning')
    %     test = (indices == ii); train = ~test; % 5=train, 1=test
    train=[]; test=[];
    t0=[];t1=[];tt0=[];tt1=[];tn0=[];tn1=[];
    
    t0=(indices1==ii);tt0=hs1(t0);tn0=hs1(~t0);
    t1=(indices2==ii);tt1=hs2(t1);tn1=hs2(~t1);
    
    test=sort([tt0' tt1']);
    train=sort([ tn0' tn1']);
    
    Dtrain=HS_feature(train,:);
    Dtest=HS_feature(test,:);
    
    Y=HS(train);
    BB = TreeBagger(ntree,Dtrain,Y,'oobpred','on');
         figure,plot(oobError(BB))
         xlabel('number of grown trees')
         ylabel('out-of-bag classification error')
   
    disp('testing')
    pred_labl=predict(BB,Dtest);
    %            [pred_labl, votes, prediction_per_tree] = classRF_predict(Dtest,BB);%for C compiler
    %pred_all_T(1,test)=str2double(pred_labl);
    %classperf(cp,str2double(pred_labl),test);
    %d(ii)=cp.CorrectRate
    toc
end


%% Making Confusion Matrix

C_M=zeros(2,2);

for ii=1:length(pred_all_T)
    %     if pred_all_T(ii)==0
    %         pred_all_T(ii)=1;
    %     end
    C_M(pred_all_T(ii),HS(ii))=C_M(pred_all_T(ii),HS(ii))+1;
end


C_M
C_M2=[C_M;sum(C_M)]
C_M3=[C_M2  sum(C_M2')']
for jj=1:2
    Sensitivity(jj)=C_M3(jj,jj)*100/C_M3(7,jj);
end


%% Making Decisions

for i=1:2
    
    TP(i)=C_M3(i,i);
    FN(i)=C_M3(7,i)-C_M3(i,i);
    FP(i)=C_M3(i,7)-C_M3(i,i);
    TN(i)=C_M3(7,7)-C_M3(i,7)-C_M3(7,i)+C_M3(i,i);
    
    ACCURACY(i)=(TP(i)+TN(i))*100/(TP(i)+FN(i)+FP(i)+TN(i));
    SENSITIVITY(i)=TP(i)*100/(TP(i)+FN(i));
    PRECISION(i)=TP(i)*100/(FP(i)+TP(i));
    SPECIFICITY(i)=TN(i)*100/(TN(i)+FP(i));
end

over_all_accuracy=trace(C_M)*100/C_M3(7,7)
toc


