function r = IJKL_KL_sym(a,b)
N = size(a,1);
r = sym(zeros(N));
for i = 1:N
    for j = 1:N
        for k = 1:N
            for l = 1:N
                r(i,j) = r(i,j)+a(i,j,k,l)*b(k,l);
            end
        end
    end
end
end