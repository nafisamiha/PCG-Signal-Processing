function [markers,record,hdr] = datareading( ki)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
 if ki==1
        load sc4002e0_hypm.mat
        markers=val;
        [hdr, record] = edfread('sc4002e0.rec');
    elseif ki==2
        load sc4012e0_hypm.mat
        markers=val;
        [hdr, record] = edfread('sc4012e0.rec');
      
    elseif ki==3
        load sc4102e0_hypm.mat
        markers=val;
       [hdr, record] = edfread('sc4102e0.rec');
        
    elseif ki==4
        load sc4112e0_hypm.mat
        markers=val;
       [hdr, record] = edfread('sc4112e0.rec');
        
    elseif ki==5
        load st7022j0_hypm.mat
        markers=val;
        [hdr, record] = edfread('st7022j0.rec');
       
    elseif ki==6
        load st7052j0_hypm.mat
        markers=val;
        [hdr, record] = edfread('st7052j0.rec');
       
    elseif ki==7
        load st7121j0_hypm.mat
        markers=val;
       [hdr, record] = edfread('st7121j0.rec');

    elseif ki==8
        load st7132j0_hypm.mat
        markers=val;
        [hdr, record] = edfread('st7132j0.rec');
        
    end
markers=markers-122;
end

