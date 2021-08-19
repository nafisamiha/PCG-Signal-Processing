clf;clc;close all; clear all;tic;

sub=[1:8];
save_frame=[];
save_marker=[];
for ki=sub
    subjectno=ki
    [markers,recrd,hdr] = datareading( ki);
    disp('Data Readed');
    %     len_marker(ki)=length(markers);
    frm_no=floor(size(recrd,2)/3000);
    %     len_frm(ki)=frm_no;
    for i=1:frm_no
        EOG=recrd(3,(i-1)*3000+1:i*3000);
        FT=abs(fft(EOG,128));
        pxx = periodogram(FT);
        %[AA,EE,KK] = aryule(EOG,4);
        if markers(i)<=6
            %             save_frame=[save_frame; mean( EOG) abs(fft(EOG,64)) ];
            save_marker=[save_marker;markers(i)];
        end
    end
    toc
end
toc
size(save_marker)
size(save_frame)
Sleep_feat=save_frame;
Sleep_stage=save_marker;

tic

