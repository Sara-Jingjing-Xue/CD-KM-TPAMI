
function [Y_label, iter_num, obj_max] = CDKM_fast(X, label,c)
% Input
% X d*n data
% label is initial label n*1
% c is the number of clusters
%  F. Nie, J. Xue, D. Wu, R. Wang, H. Li, and X. Li, 
% code for "Coordinate descent method for k-means" IEEE Transactions on Pattern Analysis and Machine Intelligence
% Output
% Y_label is the label vector n*1
% obj_max is the objective function value (max)
% iter_num is the number of iteration


[~,n] = size(X);
Y = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix 
last = 0;
iter_num = 0;

%% store once
aa=sum(Y,1);
[~,label]=max(Y,[],2);
BBB=2*(X'*X);
XX=diag(BBB)./2;

BBUU= BBB* Y;% BBUU(i,:) 
ybby=diag(Y'*BBUU/2);
%% compute Initial objective function value
   obj_max(1) = sum(ybby./aa') ; % max
while any(label ~= last)   
    last = label;       
 for i = 1:n   
     m = label(i) ;
    if aa(m)==1
        continue;  
    end 
        V21=ybby'+(BBUU(i,:)+XX(i)).*(1-Y(i,:));
        V11=ybby'-(BBUU(i,:)-XX(i)).*Y(i,:);
        delta= V21./(aa+1-Y(i,:))-V11./(aa-Y(i,:));  
    [~,q] = max(delta);     
    if m~=q                
        aa(q)= aa(q) +1; %  YY(p,p)=Y(:,p)'*Y(:,p);
        aa(m)= aa(m) -1; %  YY(m,m)=Y(:,m)'*Y(:,m)    
        ybby(m)=V11(m); %
        ybby(q)=V21(q);
        Y(i,m)=0;
        Y(i,q)=1;
        label(i)=q;
        BBUU(:,m)=BBUU(:,m)-BBB(:,i);% 
        BBUU(:,q)=BBUU(:,q)+BBB(:,i);    
    end          
 end 
  iter_num = iter_num+1;
%% compute objective function value
    obj_max(iter_num+1) = sum(ybby./aa') ; % max
end    
Y_label=label;
