
function [Y, minO, iter_num, obj] = CDKM(X, label,c)
% Input
% X d*n data
% label is initial label n*1
% c is the number of clusters
% code for F. Nie, J. Xue, D. Wu, R. Wang, H. Li, and X. Li, 
%¡°Coordinate descent method for k-means,¡± IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021
% Output
% Y is the label vector n*1
% minO is the Converged objective function value
% iter_num is the number of iteration
% obj is the objective function value


[~,n] = size(X);
F = sparse(1:n,label,1,n,c,n);  % transform label into indicator matrix 
last = 0;
iter_num = 0;
%% compute Initial objective function value
for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2); 
        center(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi;
        sumd(ii,1) = sum(d2c); 
end
 obj(1)= sum(sumd);    % Initial objective function value
%% store once
for i=1:n
    XX(i)=X(:,i)'* X(:,i);
end    
BB = X*F;
aa=sum(F,1);% diag(F'*F) ;
FXXF=BB'*BB;% F'*X'*X*F;
while any(label ~= last)   
    last = label;       
 for i = 1:n   
     m = label(i) ;
    if aa(m)==1
        continue;  
    end 
    for k = 1:c        
        if k == m   
           V1(k) = FXXF(k,k)- 2 * X(:,i)'* BB(:,k) + XX(i);
           delta(k) = FXXF(k,k) / aa(k) - V1(k) / (aa(k) -1); 
        else  
           V2(k) =(FXXF(k,k)  + 2 * X(:,i)'* BB(:,k) + XX(i));
           delta(k) = V2(k) / (aa(k) +1) -  FXXF(k,k)  / aa(k); 
        end         
    end  
    [~,q] = max(delta);     
    if m~=q        
         BB(:,q)=BB(:,q)+X(:,i); % BB(:,p)=X*F(:,p);
         BB(:,m)=BB(:,m)-X(:,i); % BB(:,m)=X*F(:,m);
         aa(q)= aa(q) +1; %  FF(p,p)=F(:,p)'*F(:,p);
         aa(m)= aa(m) -1; %  FF(m,m)=F(:,m)'*F(:,m)
         FXXF(m,m)=V1(m); 
         FXXF(q,q)=V2(q);
         label(i)=q;
    end
 end 
  iter_num = iter_num+1;
%% compute objective function value
   for ii=1:c
        idxi = find(label==ii);
        Xi = X(:,idxi);     
        ceni = mean(Xi,2);   
        center1(:,ii) = ceni;
        c2 = ceni'*ceni;
        d2c = sum(Xi.^2) + c2 - 2*ceni'*Xi; 
        sumd(ii,1) = sum(d2c); 
    end
    obj(iter_num+1) = sum(sumd) ;     %  objective function value     
end    
 minO=min(obj);
 Y=label;
end
