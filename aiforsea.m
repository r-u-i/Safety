% Ensure all the *.csv files are in the same folder as this *.m file, rather than their original folders
% This code runs well in MatLab R2019a
% Here I applied the soft margin SVM with a hybrid kernel

f0=readmatrix('part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f1=readmatrix('part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f2=readmatrix('part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f3=readmatrix('part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f4=readmatrix('part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f5=readmatrix('part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f6=readmatrix('part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f7=readmatrix('part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f8=readmatrix('part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
f9=readmatrix('part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv');
feature=[f0;f1;f2;f3;f4;f5;f6;f7;f8;f9];
label=readmatrix('part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv');
[fRow fCol]=size(feature);
[lRow lCol]=size(label);

n=0;
for i=1:lRow
    for j=1:fRow
	    if label(i,1)==feature(j,1)
		    n=n+1;
			T(n,:)=feature(j,2:end);
			L(n,1)=label(i,2);
		end
	end
end

[T,settings]=mapstd(T);
L=L.*2-ones(size(L));
train_data=T(1:round(n*0.6),:);
train_label=L(1:round(n*0.6),1);
test_data=T((round(n*0.6)+1):end,:);
test_label=L((round(n*0.6)+1):end,1);

[trainRow trainCol]=size(train_data);
[testRow testCol]=size(test_data);
Tol=1e-5;

C = [min(abs(T(:,1))),mean(abs(T(:,1))),max(abs(T(:,1)))];
for p=1:4
    for k=1:3
        for i=1:trainRow
            for j=1:trainRow
                K(i,j)=exp(-(train_data(i,:)*train_data(j,:)'-1)^p);
	        	H(i,j)=train_label(i)*train_label(j)*K(i,j);
            end
        end
        if all(eig(K))>=0
            disp('Kernel satisfies Mercer condition')
        else 
            disp('Kernel does not satisfy Mercer condition')
        end
        Aeq=train_label';
        beq=0;
        f=-ones(trainRow,1);
        lb=zeros(trainRow,1);
        ub=ones(trainRow,1)*C(k);
        options=optimset('LargeScale','off','MaxIter',10000);
        [alpha,fval,exitflag]=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

        m=0;
        b=0;
        for i=1:trainRow
            if alpha(i)>Tol 
                index=i;
                b=b+train_label(index)-(alpha(index).*train_label(index))'*exp(-(train_data(index,:)*train_data(index,:)'-1).^p);
                m=m+1;
            end
        end
        bo=b/m;

        for i = 1:trainRow
            g_train_data(i) = (alpha(i).*train_label(i))'*(exp(-(train_data(i,:)*train_data(i,:)'-1).^p))+bo;
        end

        count=0;
        for i=1:trainRow
            output=sign(g_train_data(i));
            if output-train_label(i)==0
                count=count+1;
            end
        end
        train_accuracy(p,k)=count/trainRow;     

        for i = 1:testRow
            g_test_data(i) = (alpha(i).*train_label(i))'*(exp(-(train_data(i,:)*test_data(i,:)'-1).^p))+bo;
        end

        count=0;
        for i=1:testRow
            output=sign(g_test_data(i));
            if output-test_label(i)==0
                count=count+1;
            end
        end
        test_accuracy(p,k)=count/testRow;
    end
end
train_accuracy
test_accuracy
